import pandas as pd
import math
import numpy as np

bn_map={}
hi_map={}
mr_map={}
or_map={}
pa_map={}
gu_map={}
kn_map={}
df=pd.read_csv('iscii.csv')
for i,j in zip(df.bn,df.iscii):
  if(i==i):
    bn_map[i]=j
for i,j in zip(df.dv,df.iscii):
  if(i==i):
    hi_map[i]=j
    mr_map[i]=j
for i,j in zip(df.gu,df.iscii):
  if(i==i):
    gu_map[i]=j
for i,j in zip(df.pa,df.iscii):
  if(i==i):
    pa_map[i]=j
for i,j in zip(df.kn,df.iscii):
  if(i==i):
    kn_map[i]=j
for i,j in zip(df.as,df.iscii):
  if(i==i):
    or_map[i]=j
