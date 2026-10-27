"""Sequence-aware sensitivity check for Reviewer 2.3.

Tests the one-time null A_t independent of Sp_t conditional on S_t.
It does not test causal influence or change the conditioning set.
The dependent-data reference is an asymptotic, null-centred circular
moving-block bootstrap of conditional-independence cell residuals.
It is NOT an exact finite-sample permutation test.

Prefer block_ci_test(S, A, Y, ...) inside the existing analysis pipeline.
The optional CLI is an example adapter, not a prescribed storage format:
python serial_ci.py symbols.csv --blocks 8 16 32 --draws 1999
Only that adapter expects record,side,t,S,A,Sp. One group is one uninterrupted run.
Reuse the original symbolization; no message text or data transfer is required.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np


def ci_residual(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D(p)=p_say-p_sa*p_sy/p_s; works on (...,S,A,Y) arrays."""
    ps = p.sum(axis=(-1,-2), keepdims=True)
    pa = p.sum(axis=-1, keepdims=True)
    py = p.sum(axis=-2, keepdims=True)
    expected = np.divide(pa*py, ps, out=np.zeros_like(p), where=ps>0)
    return p-expected, expected


def entropy_counts(counts: np.ndarray, corrected: bool = False) -> float:
    c=np.asarray(counts,dtype=float).ravel(); c=c[c>0]
    if not len(c): return 0.0
    n=c.sum();p=c/n; h=float(-(p*np.log2(p)).sum())
    if corrected: h+=(len(c)-1)/(2*n*np.log(2))
    return h


def contribution(counts: np.ndarray, corrected: bool = True) -> dict[str,float]:
    hS=entropy_counts(counts.sum((1,2)),corrected)
    hSA=entropy_counts(counts.sum(2),corrected)
    hSY=entropy_counts(counts.sum(1),corrected)
    hSAY=entropy_counts(counts,corrected)
    room=hSY-hS;I=room-(hSAY-hSA)
    return {'eps':float(I/room) if room>0 else float('nan'),
            'I':I,'room':room,'residual_entropy':hSAY-hSA}


def encode(S: np.ndarray,A: np.ndarray,Y: np.ndarray):
    arrays=[np.asarray(x) for x in (S,A,Y)]
    if any(x.ndim!=1 for x in arrays) or len({len(x) for x in arrays})!=1:
        raise ValueError('S,A,Sp must be nonempty aligned one-dimensional arrays.')
    if len(arrays[0])<4: raise ValueError('At least four transitions are needed.')
    encoded=[np.unique(x,return_inverse=True)[1] for x in arrays]
    sizes=tuple(int(x.max()+1) for x in encoded)
    s,a,y=encoded; code=(s*sizes[1]+a)*sizes[2]+y
    counts=np.bincount(code,minlength=int(np.prod(sizes))).reshape(sizes)
    return encoded,sizes,code,counts


def block_ci_test(S: np.ndarray,A: np.ndarray,Y: np.ndarray,*,
                  block_length: int,draws: int=1999,seed: int=0,
                  batch_size: int=200) -> dict[str,Any]:
    _,dims,code,counts=encode(S,A,Y);n=len(code);ell=int(block_length)
    if not 1<=ell<n: raise ValueError('block_length must be >=1 and < number of transitions.')
    if draws<19: raise ValueError('Use at least 19 bootstrap draws; 1999 recommended for data.')
    ph=counts.astype(float)/n;dh,p0=ci_residual(ph);active=p0>0
    if dims[1]<2 or dims[2]<2:
        return {'N':n,'block_length':ell,'status':'degenerate_marginal','p_block':float('nan')}
    statistic=float(n*np.sum(dh[active]**2/p0[active]))
    # Circular blocks of whole recorded triplets. Never form new transitions
    # between resampled blocks; each row is already a complete transition.
    d=int(np.prod(dims));onehot=np.eye(d,dtype=np.int64)[code]
    prefix=np.vstack([np.zeros((1,d),dtype=np.int64),
                      np.cumsum(np.vstack([onehot,onehot]),axis=0)])
    full,tail=divmod(n,ell);rng=np.random.default_rng(seed)
    all_stats=[]
    for left in range(0,draws,batch_size):
        b=min(batch_size,draws-left)
        starts=rng.integers(0,n,size=(b,full))
        bc=(prefix[starts+ell]-prefix[starts]).sum(axis=1)
        if tail:
            ss=rng.integers(0,n,size=b);bc+=prefix[ss+tail]-prefix[ss]
        db,_=ci_residual((bc/n).reshape((b,)+dims))
        centred=(db-dh).reshape(b,-1)[:,active.ravel()]
        tb=n*np.sum(centred**2/p0[active],axis=1)
        all_stats.extend(tb.tolist())
    bs=np.asarray(all_stats)
    # Plus-one avoids zero Monte-Carlo p-values; it does not make this
    # asymptotic bootstrap into an exact finite-sample test.
    pval=float((1+np.count_nonzero(bs>=statistic))/(draws+1))
    mm=contribution(counts,True)
    plug=contribution(counts,False)
    nind=n/ell
    warnings=[]
    if nind<10: warnings.append('fewer_than_10_block_lengths_in_record')
    if n*p0[active].min()<1: warnings.append('small_expected_cell_count')
    if not np.all(active): warnings.append('inference_only_on_observed_marginal_support')
    if mm['room']<=.05: warnings.append('original_room_screen_not_met')
    return {'N':n,'kS':dims[0],'kA':dims[1],'kY':dims[2],
            'block_length':ell,'draws':draws,'seed':seed,
            'statistic':statistic,'p_block':pval,
            'bootstrap_q95':float(np.quantile(bs,.95)),
            'eps_MM':mm['eps'],'I_MM':mm['I'],'room_MM':mm['room'],
            'I_plugin':plug['I'],'blocks_equivalent':nind,
            'minimum_expected_cell_count':float((n*p0[active]).min()),
            'status':'warnings' if warnings else 'computed',
            'warnings':';'.join(warnings)}


def _seed(*parts: object) -> int:
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16)


def analyse_csv(path: Path,blocks: list[int],draws: int,seed: int,output: Path) -> None:
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        need={'record','side','t','S','A','Sp'}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f'Input requires {sorted(need)}.')
        groups:dict[tuple[str,str],list[dict[str,str]]]={}
        for r in reader: groups.setdefault((r['record'],r['side']),[]).append(r)
    if not groups: raise ValueError('No records in input.')
    rows=[]
    for (record,side),items in sorted(groups.items()):
        items.sort(key=lambda x:int(x['t']))
        times=np.array([int(r['t']) for r in items])
        if len(np.unique(times))!=len(times) or np.any(np.diff(times)!=1):
            raise ValueError(f'{record}/{side}: t must be unique and consecutive. Do not close gaps or join runs.')
        S,A,Y=[np.array([r[c] for r in items]) for c in ('S','A','Sp')]
        for ell in blocks:
            if ell>=len(S):
                rows.append({'record':record,'side':side,'N':len(S),'block_length':ell,'status':'too_short'})
                continue
            res=block_ci_test(S,A,Y,block_length=ell,draws=draws,
                              seed=_seed(seed,record,side,ell))
            rows.append({'record':record,'side':side,**res})
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} record/side/block-length results to {output}')
    print('Report all prespecified lengths, including lost eligibility. Do not choose a length for its p-value.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--blocks',nargs='+',type=int,default=[8,16,32])
    parser.add_argument('--draws',type=int,default=1999)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--output',type=Path,default=Path('serial_ci_results.csv'))
    args=parser.parse_args()
    analyse_csv(args.input,args.blocks,args.draws,args.seed,args.output)
