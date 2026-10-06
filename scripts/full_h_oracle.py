"""Independent Slater-Condon matrix for fixed-orbital Hubbard contact U.

Only used for small validation supports; never allocated for large M.
"""
import numpy as np
from d01_operator import hop_one


def occupied(b,n):return [i for i in range(n) if (int(b)>>i)&1]


def full_h_matrix(alpha,beta,h,eri):
    m=len(alpha);n=len(h)
    if m>1000:raise ValueError('dense oracle limited to small validation supports')
    a=list(map(int,alpha));b=list(map(int,beta));oa=[occupied(x,n) for x in a];ob=[occupied(x,n) for x in b]
    out=np.zeros((m,m))
    def change(old,new):
        removed=old&~new;added=new&~old
        if bin(removed).count('1')!=1 or bin(added).count('1')!=1:return None
        src=removed.bit_length()-1;dst=added.bit_length()-1
        return src,dst,hop_one(old,src,dst)[1]
    for j in range(m):
        out[j,j]=sum(h[p,p] for p in oa[j]+ob[j])+sum(eri[p,p,q,q] for p in oa[j] for q in ob[j])
        for i in range(j):
            ca=change(a[j],a[i]);cb=change(b[j],b[i]);v=0.
            if a[i]==a[j] and cb:
                q,s,sg=cb;v=sg*(h[s,q]+sum(eri[p,p,s,q] for p in oa[j]))
            elif b[i]==b[j] and ca:
                p,r,sg=ca;v=sg*(h[r,p]+sum(eri[r,p,q,q] for q in ob[j]))
            elif ca and cb:
                p,r,sg=ca;q,s,tg=cb;v=sg*tg*eri[r,p,s,q]
            out[i,j]=out[j,i]=v
    return out
