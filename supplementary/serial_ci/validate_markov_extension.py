"""Calibration of the stronger asymptotic justification, not C1 reanalysis.
The target is first-order Markov under H0. The action retains additional memory.
Conditional contribution is exactly zero, but the action array is not generally
exchangeable conditional on the state path.
"""
import csv,json,time
from pathlib import Path
import numpy as np
from validate_serial_ci import binary_chain,permutation_p,wilson


def error_chain(n,rng):
    # Stationary Pr(E=1)=.2; second eigenvalue .9. Initialize at stationarity.
    e=np.empty(n,dtype=int);e[0]=int(rng.random()<.2)
    for t in range(1,n):
        e[t]=int(rng.random() < (.92 if e[t-1] else .02))
    return e


def main(R=500,B=399):
    rows=[];detail=[]
    for N in [250,800]:
        for scenario in ['independent_persistent_action','state_dependent_persistent_action']:
            nrej=0;valid=0;t0=time.time()
            for rep in range(R):
                seed=120000000+N*10000+(scenario.startswith('state'))*1000+rep
                rng=np.random.default_rng(seed)
                path=binary_chain(N+1,.9,rng);S,Y=path[:-1],path[1:]
                if scenario=='independent_persistent_action': A=binary_chain(N,.95,rng)
                else:A=S ^ error_chain(N,rng)
                if any(len(np.unique(x))<2 for x in (S,A,Y)):continue
                p=permutation_p(S,A,Y,B,np.random.default_rng(seed+100000000))
                valid+=1;nrej+=int(p<=.05)
                detail.append({'N':N,'scenario':scenario,'replicate':rep,'seed':seed,'p_permutation':p})
            row={'N':N,'scenario':scenario,'replicates':R,'resamples':B,'valid':valid,
                 'rejected':nrej,'rejection_rate':nrej/valid,'wilson95':wilson(nrej,valid)}
            rows.append(row);print(row,'seconds',time.time()-t0,flush=True)
    root=Path(__file__).resolve().parent/'results'
    (root/'markov_target_extension.json').write_text(json.dumps(rows,indent=2))
    with (root/'markov_target_extension_replicates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)

if __name__=='__main__': main()
