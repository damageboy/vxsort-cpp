#!/usr/bin/env python3
import sys
import pandas as pd
import plotly.express as px
import argparse


def make_vxsort_frame(df_orig):

  df = df_orig[df_orig['name'].str.startswith('BM_vxsort<')]

  df = pd.concat([df, df['name'].str.extract(r'BM_vxsort<(?P<type>[^,]+), vm::(?P<vm>[^,]+), (?P<unroll>\d+)>.*/(?P<len>\d+)/')], axis="columns")
  df = df.astype({"unroll": int}, errors='raise')
  df = df.astype({"len": int}, errors='raise')
  return df


def plot_vxsort_frame(df):
    fig = px.line(df, x='len', y='rdtsc-cycles/N', color='type', symbol='vm',
              width=1000, height=600,
              log_x=True,
              template='plotly_dark')


    fig.update_layout(title={'text': "vxsort full-sorting",
                             'x': 0.5, 'y': 0.95,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    return fig

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
parser.add_argument('-o', '--output', default=sys.stdout.buffer)
args = parser.parse_args()

df = pd.read_csv(args.filename)

if args.mode == 'vxsort-types':
  fig = plot_vxsort_frame(make_vxsort_frame(df)) 

fig.write_image(args.output, format=args.format,  engine="kaleido")
