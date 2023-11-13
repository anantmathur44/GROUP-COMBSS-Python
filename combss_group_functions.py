#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%%
## Required packages
import numpy as np
from numpy.linalg import inv, norm
from scipy.sparse.linalg import cg
from sklearn import metrics
import scipy as sc
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import block_diag
from scipy.linalg import cholesky

##Function to generate beta0 with group structure

def gen_beta0gr(p, k0, m, vrank, beta_type):
    if k0 > p:
        print("Error: k0 is greater than p")
        print(sys.path)
        sys.tracebacklimit = 1
        raise ValueError()
        
    beta0 = np.zeros(p)
    if beta_type == 1:
        gap = int(p/k0)
        for j in range(int(m)):
            beta0[j*gap*vrank:j*gap*vrank+vrank] = 1 
        return beta0
    
    elif beta_type == 2:
        S0 = np.arange(k0)
        beta0[S0] = 1
        return beta0


## Functions to convert t to w and w to a t. These are important in converting the constraint problem to an unconstraint one.

def w_to_t(w,a):
    t =  1 - np.exp(-a*w*w)
    return t

def t_to_w(t,a):
    w = np.sqrt((-np.log(1 - t)/a))
    return w

##Function to convert m dimensional vector to p dimension vector repeating entries m times

def tm_to_tp(tm, m, vrank, p):
    tp = np.zeros(p)
    for j in range(m):
        tp[j*vrank:(j+1)*vrank] = tm[j]
    return tp

## Function to generate exponential grid.
def gen_lam_grid_exp(lam_max, size, para):
    
    lam_max = norm(y)**2/(y.shape[0])  
    lam_grid = lam_max*np.array([para**i for i in range(size)])
    # lam_grid = np.append(lam_grid, 0.0)
    lam_grid = np.flip(lam_grid)
    
    return lam_grid


#Function that peforms matrix multiplication with vector v and Xall

def XtX_mv(v, Xall):
    return Xall.T@(Xall@v)



#Function that peforms matrix multiplication with vector v and Lt
# Here d is the diagonal delta in LT, tp is is p dimensional vector t. ldr is ridge lambda penalty value.

def Lt_mv(v, Xall, d, tp,N,ldr):
    return (tp*(XtX_mv((tp*v), Xall))+d*(1-(tp**2))*v+ldr*v)/N


#Function that computes gradient of obj given Xall (full design matrix), vrank (group size), m (number of groups),
#ld (lambda for l1 penalty), ldr (lambda for l2 penalty), d (delta value either a vector or scalar).

def grad_explicit(y,Xall, N, p, m, vrank, tm ,ld, ldr, d):
    #d = N
    tp = tm_to_tp(tm, m, vrank, p)
    LtO = LinearOperator((p,p), matvec= lambda v: Lt_mv(v, Xall, d, tp,N,ldr))
    tht, _ = cg(LtO,((tp*(Xall.T@y))/N))
    gammat = tp*tht;
    a = Xall.T@((Xall/N)@gammat)-(Xall.T@y/N);
    b = a - ((d-ldr)/N)*gammat;
    c, _ = cg(LtO,(tp*a))
    e = (XtX_mv((tp*c), Xall)-((d-ldr)*(tp*c)))/N
    grad = np.zeros(m)
    for j in range(m):
        thtj = tht[j*vrank:(j+1)*vrank]
        adj = a[j*vrank:(j+1)*vrank] - e[j*vrank:(j+1)*vrank]
        bj = b[j*vrank:(j+1)*vrank]
        cj = c[j*vrank:(j+1)*vrank]
        grad[j] = thtj.T@adj - bj.T@cj
    gradf = 2*grad+ld    
    return gradf


## Implementation of the ADAM optimizer for best model selection.  
# Here, alpha is stepsize, ap is scaling parameter in unconstrained transformation.
def combss_adam(y, Xall, N, p, m, vrank, tm, ld, ldr, ap, alpha, max_iter): 

    ## Parameters for Termination
    gd_maxiter = 1e5
    gd_tol = 1e-4
    max_norm = True
    epoch=10
    
    #Initialization
    count_to_term = 0
    xi1 = 0.9
    xi2 = 0.999
    u = np.zeros(m)
    v = np.zeros(m)
    epsilon = 10e-8
    
    #Truncation parameter
    eta = 1e-4
    
    tm_trun = tm.copy()
    #tm_prev = tm.copy()
    active = p
    D = np.diag(Xall.T@Xall)
    tht = np.zeros(p)
    for l in range(max_iter):
        M = np.nonzero(tm)[0]
        M_trun = np.nonzero(tm_trun)[0]
        active_new = M_trun.shape[0]
        wm = t_to_w(tm,ap) 
        tm_prev = tm.copy()
        
        if active_new == 0:
            print("0 Selected groups")
            return tm, tht
            
        if active_new != active:
            ## Find the effective set by removing the columns and rows corresponds to zero t's
            Mgrindx = np.array([[vrank*i + j for j in range(vrank)] for i in M_trun])
            Mgrindx = Mgrindx.flatten()
            Xall = Xall[:, Mgrindx]
            active = active_new
            tm_trun = tm_trun[M_trun]
              
        #Gradient Calculation
        #Keep diagonal of Lt constant N -> D[Mgrindx]
        d = N
        gradf = grad_explicit(y, Xall, N, vrank*len(tm_trun), len(tm_trun), vrank, tm_trun ,ld,ldr,d)
        w_trun = wm[M]
        gradg = gradf*(ap*(2*w_trun)*np.exp(-ap*(w_trun*w_trun)));
        # End gradient calculation
        
        ##ADAM IMPLEMENTATION
        u = xi1*u[M_trun] - (1 - xi1)*gradg
        v = xi2*v[M_trun] + (1 - xi2)*(gradg*gradg) 
    
        u_hat = u/(1 - xi1**(l+1))
        v_hat = v/(1 - xi2**(l+1))
        
        w_trun = w_trun + alpha*np.divide(u_hat, epsilon + np.sqrt(v_hat)) 
        
        wm[M] = w_trun
        tm[M] = w_to_t(w_trun,ap)
        
        wm[tm <= eta] = 0.0
        tm[tm <= eta] = 0.0
        
        tm_trun = tm[M]
        
        if max_norm:
            norm_t = max(np.abs(tm - tm_prev))
            #print(norm_t)
            if norm_t <= gd_tol:
                count_to_term += 1
                #print(count_to_term)
                #print(epoch)
                if count_to_term >= epoch:
                    break
            else:
                count_to_term = 0
    ptrun =  len(tm_trun)*vrank
    tp = tm_to_tp(tm_trun, len(tm_trun), vrank, ptrun)
    LtO = LinearOperator((ptrun,ptrun), matvec= lambda v: Lt_mv(v, Xall, d, tp,N,ldr))
    tht_trun, _ = cg(LtO,((tp*(Xall.T@y))/N))
    if active < m:
        Mgrindx_final = np.array([[vrank*i + j for j in range(vrank)] for i in M])
        tht[Mgrindx_final.flatten()] = tht_trun    
    else:
        tht = tht_trun
    print("Iteration Count: " + str(l))
    return tm, tht

#Function to compute greedy selection with k groups
#Here, ldr is the lambda term for a ridge penalty
def greedy_method(k, X,vrank, y, m, ldr):
    tc = np.zeros(m)
    for ks in range(int(k)):
        indxs = np.where(tc==0)[0]
        ls = []
        print(ks)
        for i in indxs:
            tct = tc.copy()
            tct[i] = 1
            ls.append(get_error(y, X, vrank,tct,ldr))
        indx = np.where(ls == min(ls))[0][0]
        tc[indxs[indx]] = 1
    return tc

#Given a binary vector s, compute the training error
def get_error(y, X,vrank, s,ldr):
    m = len(s)
    p = m * vrank
    s = np.where(tm_to_tp(s, m, vrank, p)>0)[0]
    #print(s)
    Xs = X[:,s]
    #print(Xs.shape)
    ps = len(s)
    return (np.linalg.norm(y-Xs@np.linalg.pinv(Xs.T@Xs+ldr*np.eye(ps))@Xs.T@y))


#Given a binary vector s, ridge penalty ldr, compute the theta vector

def get_theta(y, X,vrank, s,ldr):
    m = len(s)
    p = m * vrank
    s = np.where(tm_to_tp(s, m, vrank, p)>0)[0]
    #print(s)
    Xs = X[:,s]
    print(Xs.shape)
    thetashr = np.linalg.pinv(Xs.T@Xs+ldr*np.eye(len(s)))@Xs.T@y
    theta = np.zeros(p)
    theta[s] = thetashr
    return theta
