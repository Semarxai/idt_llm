"""Independent calibration, not a reanalysis of the manuscript conversations.
Reproduce with: python validate_serial_ci.py --replicates 300 --draws 399
Data-generating laws, sample sizes, and block grid are fixed below.
All outputs, including adverse small-sample results, are retained.
"""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from serial_ci import block_ci_test


def binary_chain(n: int,p: float,rng: np.random.Generator) -> np.ndarray:
    return np.r_[rng.integers(2),(rng.random(n-1)>p).astype(int)].cumsum()%2


def state_path(n: int,p: float,order: int,rng: np.random.Generator)->np.ndarray:
    if order==1: return binary_chain(n,p,rng)
    out=np.empty(n,dtype=int)
    for parity in range(order):
        out[parity::order]=binary_chain(len(out[parity::order]),p,rng)
    return out


def h_batch(z:np.ndarray,k:int)->np.ndarray:
    z=np.atleast_2d(z);b,n=z.shape
    counts=np.bincount((z+np.arange(b)[:,None]*k).ravel(),minlength=b*k).reshape(b,k)
    p=counts/n
    h=-np.sum(p*np.log2(np.where(p>0,p,1)),axis=1)
    return h+((counts>0).sum(axis=1)-1)/(2*n*np.log(2))


def eps_batch(S,A,Y):
    A=np.atleast_2d(A)
    room=h_batch(S*2+Y,4)-h_batch(S,2)
    res=h_batch((S*2+A)*2+Y,8)-h_batch(S*2+A,4)
    return (room-res)/room


def permutation_p(S,A,Y,B,rng):
    obs=float(eps_batch(S,A,Y)[0]);aa=np.tile(A,(B,1))
    for s in np.unique(S):
        ix=np.flatnonzero(S==s)
        aa[:,ix]=rng.permuted(np.tile(A[ix],(B,1)),axis=1)
    vals=eps_batch(S,aa,Y)
    return float((1+(vals>=obs).sum())/(B+1))


def wilson(x,n,z=1.959963984540054):
    p=x/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]


def run(R:int=300,B:int=399):
    scenarios=[('iid',.5,1,.5),('state_memory_only',.95,2,.5),
               ('independent_persistent_paths_090',.9,2,.9),
               ('independent_persistent_paths_095',.95,2,.95)]
    blocks=[8,16,32]
    root=Path(__file__).resolve().parent/'results';root.mkdir(exist_ok=True)
    summary=[];detail=[]
    for N in [250,800]:
        for si,(name,ps,order,pa) in enumerate(scenarios):
            tic=time.time();pvals={key:[] for key in ['within_S_permutation']+[f'block_{l}' for l in blocks]}
            for rep in range(R):
                seed=79000000+N*10000+si*1000+rep
                rng=np.random.default_rng(seed)
                path=state_path(N+1,ps,order,rng);A=binary_chain(N,pa,rng);S,Y=path[:-1],path[1:]
                if any(len(np.unique(x))<2 for x in (S,A,Y)): continue
                pp=permutation_p(S,A,Y,B,np.random.default_rng(seed+50000000))
                pvals['within_S_permutation'].append(pp)
                row={'scenario':name,'N':N,'replicate':rep,'seed':seed,'within_S_permutation':pp}
                for ell in blocks:
                    res=block_ci_test(S,A,Y,block_length=ell,draws=B,seed=seed+ell*100000000)
                    pv=res['p_block'];pvals[f'block_{ell}'].append(pv);row[f'block_{ell}']=pv
                detail.append(row)
            for method,pv in pvals.items():
                rejected=int(np.count_nonzero(np.array(pv)<=.05));n=len(pv)
                summary.append({'scenario':name,'N':N,'method':method,
                                'rejected':rejected,'valid':n,'rejection_rate':rejected/n,
                                'wilson95':wilson(rejected,n),'replicates':R,'resamples':B})
            print(name,N,{m:round(np.mean(np.array(v)<=.05),3) for m,v in pvals.items()},
                  f'{time.time()-tic:.1f}s',flush=True)
            (root/'calibration_summary.json').write_text(json.dumps(summary,indent=2))
            with (root/'calibration_replicates.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replicates',type=int,default=300)
    p.add_argument('--draws',type=int,default=399);a=p.parse_args();run(a.replicates,a.draws)
