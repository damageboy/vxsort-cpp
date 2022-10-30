#!/usr/bin/env python3
import sys
import mmap
import humanize
import re
import pandas as pd
import plotly.express as px
import argparse


def make_vxsort_types_frame(df_orig):
  df = df_orig[df_orig['name'].str.startswith('BM_vxsort<')]

  df = pd.concat([df, df['name'].str.extract(r'BM_vxsort<(?P<type>[^,]+), vm::(?P<vm>[^,]+), (?P<unroll>\d+)>.*/(?P<len>\d+)/')], axis="columns")
  df = pd.concat([df, df['type'].str.extract(r'(?P<typecat>.)(?P<width>\d+)')], axis="columns")
  df = df.astype({"width": int}, errors='raise')
  df = df.astype({"unroll": int}, errors='raise')
  df = df.astype({"len": int}, errors='raise')
  
  return df


def make_title(title: str):
  return { 'text': title,
          'x': 0.5, 'y': 0.95,                             
          'xanchor': 'center',
          'yanchor': 'top'
         }
  

def plot_vxsort_types_frame(df):
    fig = px.line(df, x='len', y='rdtsc-cycles/N', color='type', symbol='vm',
              width=1000, height=600,
              log_x=True,
              labels={
                "len_title": "Problem size",
                "len": "Problem size",
                "rdtsc-cycles/N": "cycles per element",
              },                 
              template='plotly_dark')


    fig.update_layout(title=make_title("vxsort full-sorting"),
                      yaxis_tickangle=-30)

    return fig

def make_vxsort_vs_all_frame(df_orig):

  #df = df_orig[df_orig['name'].str.startswith('BM_vxsort<')]
  
  df = pd.concat([df_orig, df_orig['name'].str.extract(r'BM_(?P<sorter>vxsort|pdqsort_branchless|stdsort)<(?P<type>[^,]+).*>/(?P<len>\d+)/')], axis="columns")  
  df = pd.concat([df, df['name'].str.extract(r'BM_vxsort<.*vm::(?P<vm>[^,]+), (?P<unroll>\d+)>/')], axis="columns")
  df = pd.concat([df, df['type'].str.extract(r'(?P<typecat>.)(?P<width>\d+)')], axis="columns")
  df.fillna(0, inplace=True)
  df = df.astype({"width": int}, errors='raise')
  df = df.astype({"unroll": int}, errors='raise')
  df = df.astype({"len": int}, errors='raise')
    
  df['sorter_title'] = df.apply(lambda x: f"{x['sorter']}{'/' + x['vm'] if x['vm'] != 0 else ''}", axis=1)
  
  df.dropna(axis=0, subset=['sorter'], inplace=True)

  return df

def plot_vxsort_vs_all_frame(df):
  
    cardinality = df.nunique(dropna=True)
    
    if cardinality['sorter'] == 1:
      raise ValueError("Only one sorter in the frame")
    
    if cardinality['type'] > 1:
      raise ValueError("vxsort vs. all plots must have only one type")
    
    df['len_title'] = df.apply(lambda x: f"{humanize.naturalsize(x['len'], gnu=True, binary=True).replace('B', '')}", axis=1)
            
    fig = px.bar(df,
                 barmode='group',
                 orientation='h',
                 color='sorter_title',
                 y='len_title',
                 x='rdtsc-cycles/N',                                  
                 width=1000, height=600,
                 labels={
                   "len_title": "Problem size",
                   "len": "Problem size",
                   "rdtsc-cycles/N": "cycles per element",
                  },                 
                 template='plotly_dark')
    
    fig.update_layout(title=make_title("vxsort vs. others"),
                      bargap=0.3, bargroupgap=0.2,
                      yaxis_tickangle=-30,
                      )

    return fig

def parse_args():
  parser = argparse.ArgumentParser(
                      prog = 'make-figure.py',
                      description = 'Generate pretty figures for vxsort benchmarks')
                    
  parser.add_argument('filename')
  parser.add_argument('--mode',
                      choices=('vxsort-types', 'vxsort-vs-all'),
                      const='vxsort-types', 
                      default='vxsort-types',
                      nargs='?',
                      help='which figure to generate (default: %(const)s)')

  parser.add_argument('--format', choices=['svg', 'png', 'html'], default='svg')
  parser.add_argument('--query', action='append',
                      help='pandas query to filter the data-frame with before plotting')
  parser.add_argument('--debug-df', action='store_true', help='just show the last data-frame before generating a figure and quit')
  parser.add_argument('-o', '--output', default=sys.stdout.buffer)
  args = parser.parse_args()
  return args

def parse_csv_into_dataframe(filename):
  with open(filename) as f:
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for match in re.finditer(b'name,iterations,real_time,cpu_time,time_unit', m):
      f.seek(match.start())
      break
    df = pd.read_csv(f)
  
  # drop some commonly useless columns  
  df.drop(['iterations', 'real_time', 'cpu_time', 'time_unit', 'label',
           'items_per_second', 'error_occurred', 'error_message'], axis=1, inplace=True)
  return df

def apply_queries(df, queries):
  if not queries or len(queries) == 0:
    return df

  for q in queries:
      df = df.query(q, engine='python')

  return df

def make_figures():
  args = parse_args()

  df = parse_csv_into_dataframe(args.filename)

  if args.mode == 'vxsort-types': 
    plot_df = make_vxsort_types_frame(df) 
    plot_df = apply_queries(plot_df, args.query)
    fig = plot_vxsort_types_frame(plot_df) 
  elif args.mode == 'vxsort-vs-all':  
    plot_df = make_vxsort_vs_all_frame(df)
    if not args.query or len(arg.query) == 0:
      args.query = [ "len <= 1048576 & width == 32 & typecat == 'i' & (sorter != 'vxsort' | (unroll == 8))" ]
  
    plot_df = apply_queries(plot_df, args.query)
    fig = plot_vxsort_vs_all_frame(plot_df)

  if args.debug_df:
    print(plot_df)
    sys.exit()

  if args.format == 'html':
    fig.write_html(args.output)
  else:
    fig.write_image(args.output, format=args.format,  engine="kaleido")

if __name__ == "__main__":
  make_figures()
