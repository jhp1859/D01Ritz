"""Large-bank generalized Ritz iteration with explicit ridge sensitivity.

Only N x M sample banks and O(M*blocksize) vectors are stored. The ridge is
reported, and convergence is judged using the ORIGINAL unregularized pencil.
"""
import warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator,lobpcg


def solve_iterative(F,G,cutoff,maxiter=80):
    n,m=F.shape
    def S(v):return F.T@(F@v)/n
    def A(v):return (F.T@(G@v)+G.T@(F@v))/(2*n)
    rng=np.random.default_rng(643871)
    v=rng.normal(size=m);v/=np.linalg.norm(v)
    for _ in range(20):
        v=S(v);v/=np.linalg.norm(v)
    smax=float(v@S(v));ridge=cutoff*smax
    def B(v):return S(v)+ridge*v
    diagS=np.einsum('ij,ij->j',F,F)/n
    diagA=np.einsum('ij,ij->j',F,G)/n
    scale=1/np.maximum(abs(diagA)+30*diagS,1e-12)
    def pre(v):return scale*v if v.ndim==1 else scale[:,None]*v
    op=lambda f:LinearOperator((m,m),matvec=f,matmat=f,dtype=np.float64)
    X=rng.normal(size=(m,3))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        values,vectors,history=lobpcg(op(A),X,B=op(B),M=op(pre),largest=False,tol=1e-8,maxiter=maxiter,retResidualNormsHistory=True)
    c=vectors[:,np.argmin(values)];c/=np.sqrt(c@S(c));ac=A(c);sc=S(c);E=float(c@ac)
    res=float(np.linalg.norm(ac-E*sc)/(np.linalg.norm(ac)+abs(E)*np.linalg.norm(sc)))
    return c,{'energy':E,'retained_rank':None,'empirical_nullity':max(0,m-n),
        'rank_note':'rank not inferred from iterations; nullity is only the sample-count lower bound',
        'overlap_retained_condition':None,'overlap_full_condition':None,
        'regularized_condition_upper_bound':(float(diagS.sum())+ridge)/ridge if ridge>0 else None,
        'overlap_max_eigenvalue_power_estimate':smax,'cutoff':cutoff,'ridge':ridge,
        'overlap_regularization':'ridge, not hard spectral truncation',
        'norm_S_over_Z':float(c@sc),'relative_generalized_residual':res,
        'projected_residual':float(np.linalg.norm(ac-E*sc)),
        'solver':'matrix-free block LOBPCG with diagonal preconditioning; unregularized residual reported',
        'iteration_limit':maxiter,'iterations_recorded':len(history),'last_regularized_residuals':np.asarray(history[-1]).tolist(),
        'warnings':[str(w.message) for w in captured]}
