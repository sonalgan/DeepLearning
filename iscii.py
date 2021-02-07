import pandas as pd
import math
import numpy as np
values='abcdef0123456789*'
data=['a0','b1','c2*']
def iscii_encoded(data):
  char_to_int = dict((c, i) for i, c in enumerate(values))
  integer_encoded=[]
  for i in range(len(data)):
    integer_encoded.append([char_to_int[char] for char in data[i]])
  iscii_encoded = []   
  for i in integer_encoded:
    letter = np.zeros(len(values))
    for j in i:
      letter[j]=1
    iscii_encoded.append(letter)
  return np.array(iscii_encoded)
languages=["hi","bn","gu","kn","mr","or","ta","te","ml","pa"]
keys=[]
languages.sort()
lang_map={ lang:{ key:[] for key in keys} for lang in languages}
print(lang_map)
df=pd.read_csv('iscii.csv')
for lang in languages:
  '''
  for i,j in zip(df.bn,df.iscii):
  if(i==i):
    bn_map[i]=j
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
    '''