import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / 'src'))
from paths import *
import json,glob,os,pandas as pd
rows=[]
for f in sorted(glob.glob('/tmp/pb/logs_x/logs/*.json')):
    d=json.load(open(f)); gid=str(d['game_id'])
    seq=[]
    for r in d['rounds']:
        for m in r['messages']:
            sp=m['speaker']; tx=str(m['message'])
            if seq and seq[-1][0]==sp: seq[-1][1]=seq[-1][1]+' '+tx   # collapse consecutive same-speaker
            else: seq.append([sp,tx])
    gname=os.path.basename(f)[:-5]
    for i,(sp,tx) in enumerate(seq): rows.append((gname,i,sp,tx))
df=pd.DataFrame(rows,columns=['game','idx','speaker','text'])
df.to_csv(str(C3 / 'c3_messages.csv'),index=False)
g=df.groupby('game').size()
print("games:",len(g),"| >=25 msgs:",(g>=25).sum(),"| >=50:",(g>=50).sum())
print(g.describe())
