#!/usr/bin/env python3
import sys
import mmap
import humanize
import re
import pandas as pd
import plotly.express as px
import argparse
import math


def make_vxsort_types_frame(df_orig):
    df = df_orig[df_orig['name'].str.startswith('BM_vxsort<')]

    df = pd.concat(
        [df, df['name'].str.extract(
            r'BM_vxsort<(?P<type>[^,]+), vm::(?P<vm>[^,]+), (?P<unroll>\d+)>.*/(?P<len>\d+)/')],
        axis="columns")
    df = pd.concat([df, df['type'].str.extract(
        r'(?P<typecat>.)(?P<width>\d+)')], axis="columns")
    df = df.astype({"width": int}, errors='raise')
    df = df.astype({"unroll": int}, errors='raise')
    df = df.astype({"len": int}, errors='raise')

    df['len_bytes'] = df['len'] * df['width'] / 8

    return df


def make_bitonic_types_frame(df_orig):
    df = df_orig[df_orig['name'].str.startswith('BM_bitonic_sort<')]

    df = pd.concat(
        [df, df['name'].str.extract(
            r'BM_bitonic_sort<(?P<type>[^,]+), vm::(?P<vm>[^,]+)>.*/(?P<len>\d+)/')],
        axis="columns")
    df = pd.concat([df, df['type'].str.extract(
        r'(?P<typecat>.)(?P<width>\d+)')], axis="columns")
    df = df.astype({"width": int}, errors='raise')
    df = df.astype({"len": int}, errors='raise')

    df['len_bytes'] = df['len'] * df['width'] / 8

    return df


def make_title(title: str):
    return {'text': title,
            'x': 0.5, 'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
            }


def add_cache_vline(fig, cache, name, color, len_min, len_max):
    if cache < len_min or cache > len_max:
        return

    fig.add_vline(cache, line_width=2,
                  line_dash="dash",
                  line_color=color)

    fig.add_annotation(x=(math.log(cache)) / math.log(10), y=2,
                       showarrow=False,
                       xshift=-15,
                       font=dict(
                           family="sans serif",
                           size=14,
                           color=color),
                       text=name,
                       textangle=-30, )


def make_log2_ticks(min, max):
    ticks = []
    tick_labels = []
    while min <= max:
        ticks.append(min)
        tick_labels.append(humanize.naturalsize(int(min), gnu=True,
                                                binary=True).replace('B', ''))
        min *= 2
    return ticks, tick_labels


def plot_sort_types_frame(df, title, args, caches):
    fig = px.line(df,
                  x='len_bytes',
                  y='rdtsc-cycles/N',
                  color='type',
                  symbol='vm',
                  width=1000, height=600,
                  log_x=True,
                  labels={
                      "len": "Problem size",
                      "len_bytes": "Problem size (bytes)",
                      "rdtsc-cycles/N": "cycles per element",
                  },
                  template=args.template)

    len_min, len_max = df['len_bytes'].min(), df['len_bytes'].max()
    add_cache_vline(fig, caches[0], "L1", "green", len_min, len_max)
    add_cache_vline(fig, caches[1], "L2", "gold", len_min, len_max)
    add_cache_vline(fig, caches[2], "L3", "red", len_min, len_max)

    tick_values, tick_labels = make_log2_ticks(
        df['len_bytes'].min(), df['len_bytes'].max())

    fig.update_xaxes(tickvals=tick_values, ticktext=tick_labels)

    fig.update_layout(title=make_title(title),
                      yaxis_tickangle=-30)

    return fig


def make_vxsort_vs_all_frame(df_orig):
    # df = df_orig[df_orig['name'].str.startswith('BM_vxsort<')]

    df = pd.concat([df_orig, df_orig['name'].str.extract(
        r'BM_(?P<sorter>vxsort|pdqsort_branchless|stdsort)<(?P<type>[^,]+).*>/(?P<len>\d+)/')], axis="columns")
    df = pd.concat([df, df['name'].str.extract(
        r'BM_vxsort<.*vm::(?P<vm>[^,]+), (?P<unroll>\d+)>/')], axis="columns")
    df = pd.concat([df, df['type'].str.extract(
        r'(?P<typecat>.)(?P<width>\d+)')], axis="columns")
    df.fillna(0, inplace=True)
    df = df.astype({"width": int}, errors='raise')
    df = df.astype({"unroll": int}, errors='raise')
    df = df.astype({"len": int}, errors='raise')

    df['sorter_title'] = df.apply(
        lambda x: f"{x['sorter']}{'/' + x['vm'] if x['vm'] != 0 else ''}", axis=1)

    df.dropna(axis=0, subset=['sorter'], inplace=True)

    return df


def plot_vxsort_vs_all_frame(df, args):
    df['len_title'] = df.apply(
        lambda x: f"{humanize.naturalsize(x['len'], gnu=True, binary=True).replace('B', '')}", axis=1)

    cardinality = df[['len_title', 'type',
                      'sorter_title']].nunique(dropna=True)

    if cardinality['sorter_title'] == 1:
        raise ValueError("Only one sorter in the frame")

    if cardinality['type'] == 1 and cardinality['len_title'] > 1:
        title_suffix = f"({df['type'].unique()[0]})"
        y_column = 'len_title'
    elif cardinality['type'] > 1 and cardinality['len_title'] == 1:
        title_suffix = f"({df['len_title'].unique()[0]} elements)"
        y_column = 'type'
    else:
        raise ValueError(
            f"Can't figure out the comparison axis for the plot: {cardinality}")

    if args.speedup:
        baseline_df = df[df['sorter_title'] == args.speedup]
        df['speedup'] = df.groupby(y_column)['rdtsc-cycles/N']. \
            transform(lambda x: baseline_df[baseline_df[y_column]
                                            == x.name]['rdtsc-cycles/N'].values[0] / x)
        x_column = 'speedup'
    else:
        x_column = 'rdtsc-cycles/N'

    df.sort_values([x_column], ascending=[False], inplace=True)

    fig = px.bar(df,
                 barmode='group',
                 orientation='h',
                 color='sorter_title',
                 x=x_column,
                 y=y_column,
                 width=1000, height=600,
                 labels={
                     "len_title": "Problem size",
                     "len": "Problem size",
                     "sorter_title": "Sorter",
                     "rdtsc-cycles/N": "Cycles/element",
                     "speedup": f"speedup over {args.speedup}",
                 },
                 template=args.template)

    fig.update_layout(title=make_title(f"vxsort vs. others {title_suffix}"),
                      bargap=0.3, bargroupgap=0.2,
                      yaxis_tickangle=-30,
                      )
    if format == 'html':
        fig.update_layout(margin=dict(t=100, b=0, l=0, r=0))

    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        prog='make-figure.py',
        description='Generate pretty figures for vxsort benchmarks')

    parser.add_argument('filename')
    parser.add_argument('--mode',
                        choices=('vxsort-types', 'vxsort-vs-all', 'bitonic-types'),
                        const='vxsort-types',
                        default='vxsort-types',
                        nargs='?',
                        help='which figure to generate (default: %(const)s)')

    parser.add_argument(
        '--format', choices=['svg', 'png', 'html'], default='svg')
    parser.add_argument('--query', action='append',
                        help='pandas query to filter the data-frame with before plotting')
    parser.add_argument(
        '--speedup', help='plot speedup vs. supplied baseline sorter')
    parser.add_argument('--debug-df', action='store_true',
                        help='just show the last data-frame before generating a figure and quit')
    parser.add_argument('-o', '--output', default=sys.stdout.buffer)
    parser.add_argument('--template', default='plotly_dark')
    args = parser.parse_args()
    return args


def parse_cache_tidbit(cache_type, text):
    m = re.search(cache_type + ' (\d+) (KiB|MiB)', text)
    if m:
        cachesize = int(m.group(1))
        unit = m.group(2)
        cachesize *= 1024 if unit == 'KiB' else 1024 * 1024
        return cachesize
    return None


def parse_csv_into_dataframe(filename):
    with open(filename) as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for match in re.finditer(b'name,iterations,real_time,cpu_time,time_unit', m):
            header = f.read(match.start())
            f.seek(match.start())
            break

        l1d_size = parse_cache_tidbit('L1 Data', header)
        l2_size = parse_cache_tidbit('L2 Unified', header)
        l3_size = parse_cache_tidbit('L3 Unified', header)

        df = pd.read_csv(f)

    # drop some commonly useless columns
    df.drop(['iterations', 'real_time', 'cpu_time', 'time_unit', 'label',
             'items_per_second', 'error_occurred', 'error_message'], axis=1, inplace=True)
    return ((l1d_size, l2_size, l3_size), df)


def apply_queries(df, queries):
    if not queries or len(queries) == 0:
        return df

    for q in queries:
        df = df.query(q, engine='python')

    return df


def make_figures():
    args = parse_args()

    caches, df = parse_csv_into_dataframe(args.filename)

    if args.mode == 'vxsort-types':
        if args.speedup:
            raise argparse.ArgumentError(
                "Speedup mode is not supported for vxsort-types mode")
        plot_df = make_vxsort_types_frame(df)
        plot_df = apply_queries(plot_df, args.query)
        fig = plot_sort_types_frame(plot_df, "vxsort full-sorting", args, caches)
    elif args.mode == 'vxsort-vs-all':
        plot_df = make_vxsort_vs_all_frame(df)
        if not args.query or len(args.query) == 0:
            args.query = [
                "len <= 1048576 & width == 32 & typecat == 'i' & (sorter != 'vxsort' | unroll == 8)"]

        plot_df = apply_queries(plot_df, args.query)
        fig = plot_vxsort_vs_all_frame(plot_df, args)
    elif args.mode == 'bitonic-types':
        plot_df = make_bitonic_types_frame(df)
        plot_df = apply_queries(plot_df, args.query)
        fig = plot_sort_types_frame(plot_df, "vxsort bitonic-sorting", args, caches)

    if args.debug_df:
        print(plot_df)
        sys.exit()

    if args.format == 'html':
        fig.write_html(args.output)
    else:
        fig.write_image(args.output, format=args.format, engine="kaleido")


if __name__ == "__main__":
    make_figures()
