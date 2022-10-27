#!/usr/bin/env python3

import pandas as pd
import plotly.express as px

df = pd.read_csv("figure.csv", skiprows=10)
df['type']=df['name'].str.extract(r'BM_vxsort<([^,]*),', expand=True)
df['vm']=df['name'].str.extract(r'vm::([^,]*),', expand=True)
df['unroll']=df['name'].str.extract(r', (\d+)>', expand=True)
df['len']=df['name'].str.extract(r'/(\d+)/', expand=True)
df = df.astype({"unroll": int}, errors='raise')
df = df.astype({"len": int}, errors='raise')

fig = px.line(df, x='len', y='rdtsc-cycles/N', color='type', symbol='vm',
              width=1000, height=600,
              log_x=True,
              template='plotly_dark')


fig.update_layout(title={ 'text': "vxsort full-sorting",
                          'x': 0.5, 'y': 0.95,
                          'xanchor': 'center',
                          'yanchor': 'top'})


fig.write_image("vxsort-full.png", engine="kaleido")
fig.write_html("vxsort-full.html")

