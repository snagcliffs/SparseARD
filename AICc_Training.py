import numpy as np
from SparseARD import *

tol=1e-5

def Train_ARDr(Theta, y, ARD_results=None, lams = None, eta = None, verbose = False):
    
    if lams is None: lams = [np.linalg.norm(y)**2*l for l in [0,1e-3,1e-2,1e-1,1e0,1e1,1e2]] 
    if eta is None: eta = 0.1
    if ARD_results is None:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,_ = SBL([Theta,y], \
                                       sigma2=1, \
                                       estimate_sigma=True, \
                                       maxit=500, \
                                       verbose=verbose, \
                                       tol=tol)
        AICc_ARD = AICc(Theta,y,gamma_ARD,sigma2_ARD)
    else:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,AICc_ARD = ARD_results

    Gamma_ARDr = [gamma_ARD]
    Sigma2_ARDr = [sigma2_ARD]
    Mu_xi_ARDr = [mu_xi_ARD]
    AICc_ARDr = [AICc_ARD]

    for lam in lams[1:]:

        if verbose: print('Training with lambda='+str(lam), end = '. ')

        gamma,sigma2,mu_xi,_ = SBL([Theta,y], \
                                 gamma=gamma_ARD, \
                                 lam=lam, \
                                 eta=eta, \
                                 sigma2=sigma2_ARD, \
                                 maxit = 500, \
                                 tol = tol, \
                                 verbose = verbose, \
                                 estimate_sigma=True, \
                                 regularize = True)

        Gamma_ARDr.append(gamma)
        Sigma2_ARDr.append(sigma2)
        Mu_xi_ARDr.append(mu_xi)
        AICc_ARDr.append(AICc(Theta,y,gamma,sigma2))
        if verbose: print('AIC_c='+str(AICc_ARDr[-1]))

    q = np.argmin(AICc_ARDr)
    gamma_ARDr = Gamma_ARDr[q]
    sigma2_ARDr = Sigma2_ARDr[q]
    mu_xi_ARDr = Mu_xi_ARDr[q]
    
    return gamma_ARDr, sigma2_ARDr, mu_xi_ARDr

def Train_ARDvi(Theta, y, ARD_results=None, alphas = None, verbose = False):
    
    if alphas is None: alphas = [1,2,4,8,10,16,32,64,128]
    if ARD_results is None:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,_ = SBL([Theta,y], \
                                       sigma2=1, \
                                       estimate_sigma=True, \
                                       maxit=500, \
                                       verbose=verbose, \
                                       tol=tol)
        AICc_ARD = AICc(Theta,y,gamma_ARD,sigma2_ARD)
    else:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,AICc_ARD = ARD_results
        
    Gamma_ARDvi = [gamma_ARD]
    Sigma2_ARDvi = [sigma2_ARD]
    Mu_xi_ARDvi = [mu_xi_ARD]
    AICc_ARDvi = [AICc_ARD]

    for alpha in alphas[1:]:

        if verbose: print('Training with alpha='+str(alpha), end = '. ')

        gamma, sigma2, mu_xi, _ = SBL([Theta,y], \
                                      sigma2=1, \
                                      estimate_sigma=True, \
                                      maxit=500, \
                                      verbose=verbose, \
                                      tol=tol, \
                                      alpha = alpha)

        Gamma_ARDvi.append(gamma)
        Sigma2_ARDvi.append(sigma2)
        Mu_xi_ARDvi.append(mu_xi)
        AICc_ARDvi.append(AICc(Theta,y,gamma,sigma2))
        if verbose: print('AIC_c='+str(AICc_ARDvi[-1]))

    q = np.argmin(AICc_ARDvi)
    gamma_ARDvi = Gamma_ARDvi[q]
    sigma2_ARDvi = Sigma2_ARDvi[q]
    mu_xi_ARDvi = Mu_xi_ARDvi[q]
    
    return gamma_ARDvi, sigma2_ARDvi, mu_xi_ARDvi

def Train_M_STSBL(Theta, y, ARD_results=None, taus = None, verbose = False):
    
    if taus is None: taus = [0,1e-5,1e-4,1e-3,1e-2,1e-1,1]
    if ARD_results is None:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,_ = SBL([Theta,y], \
                                       sigma2=1, \
                                       estimate_sigma=True, \
                                       maxit=500, \
                                       verbose=verbose, \
                                       tol=tol)
        AICc_ARD = AICc(Theta,y,gamma_ARD,sigma2_ARD)
    else:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,AICc_ARD = ARD_results
    
    Gamma_STSBL = [gamma_ARD]
    Sigma2_STSBL = [sigma2_ARD]
    Mu_xi_STSBL = [mu_xi_ARD]
    AICc_STSBL = [AICc_ARD]

    for tau in taus[1:]:

        if verbose: print('Training with tau='+str(tau), end = '. ')

        mu_xi, gamma, sigma2 = M_STSBL(Theta, y, tau, tol=tol, verbose=verbose, \
                                     warm_start = [np.copy(gamma_ARD), np.copy(sigma2_ARD), np.copy(mu_xi_ARD)])

        Gamma_STSBL.append(gamma)
        Sigma2_STSBL.append(sigma2)
        Mu_xi_STSBL.append(mu_xi)
        AICc_STSBL.append(AICc(Theta,y,gamma,sigma2))
        if verbose: print('AIC_c='+str(AICc_STSBL[-1]))

    q = np.argmin(AICc_STSBL)
    gamma_STSBL = Gamma_STSBL[q]
    sigma2_STSBL = Sigma2_STSBL[q]
    mu_xi_STSBL = Mu_xi_STSBL[q]
    
    return gamma_STSBL, sigma2_STSBL, mu_xi_STSBL

def Train_L_STSBL(Theta, y, ARD_results=None, taus = None, verbose = False):
    
    if taus is None: taus = [np.inf,1e-4,1e-3,1e-2,1e-1,1,2,5,10]
    if ARD_results is None:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,_ = SBL([Theta,y], \
                                       sigma2=1, \
                                       estimate_sigma=True, \
                                       maxit=500, \
                                       verbose=verbose, \
                                       tol=tol)
        AICc_ARD = AICc(Theta,y,gamma_ARD,sigma2_ARD)
    else:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,AICc_ARD = ARD_results
    
    Gamma_L_STSBL = [gamma_ARD]
    Sigma2_L_STSBL = [sigma2_ARD]
    Mu_xi_L_STSBL = [mu_xi_ARD]
    AICc_L_STSBL = [AICc_ARD]

    for tau in taus[1:]:

        if verbose: print('Training with tau='+str(tau), end = '. ')

        mu_xi, gamma, sigma2 = L_STSBL(Theta, y, tau, tol=tol, verbose=verbose, \
                                     warm_start = [np.copy(gamma_ARD), np.copy(sigma2_ARD), np.copy(mu_xi_ARD)])

        Gamma_L_STSBL.append(gamma)
        Sigma2_L_STSBL.append(sigma2)
        Mu_xi_L_STSBL.append(mu_xi)
        AICc_L_STSBL.append(AICc(Theta,y,gamma,sigma2))
        if verbose: print('AIC_c='+str(AICc_L_STSBL[-1]))

    q = np.argmin(AICc_L_STSBL)
    gamma_L_STSBL = Gamma_L_STSBL[q]
    sigma2_L_STSBL = Sigma2_L_STSBL[q]
    mu_xi_L_STSBL = Mu_xi_L_STSBL[q]
    
    return gamma_L_STSBL, sigma2_L_STSBL, mu_xi_L_STSBL

def Train_MAP_STSBL(Theta, y, ARD_results=None, taus = None, verbose = False):
    
    if taus is None: taus = [0,1e-2,1e-1,1,1e1,1e2,1e3]
    if ARD_results is None:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,_ = SBL([Theta,y], \
                                       sigma2=1, \
                                       estimate_sigma=True, \
                                       maxit=500, \
                                       verbose=verbose, \
                                       tol=tol)
        AICc_ARD = AICc(Theta,y,gamma_ARD,sigma2_ARD)
    else:
        gamma_ARD,sigma2_ARD,mu_xi_ARD,AICc_ARD = ARD_results
    
    Gamma_d_STSBL = [gamma_ARD]
    Sigma2_d_STSBL = [sigma2_ARD]
    Mu_xi_d_STSBL = [mu_xi_ARD]
    AICc_d_STSBL = [AICc_ARD]

    for tau in taus[1:]:

        if verbose: print('Training with tau='+str(tau), end = '. ')

        mu_xi, gamma, sigma2 = MAP_STSBL(Theta, y, tau, 1, tol=tol, verbose=verbose, \
                                     warm_start = [np.copy(gamma_ARD), np.copy(sigma2_ARD), np.copy(mu_xi_ARD)])

        Gamma_d_STSBL.append(gamma)
        Sigma2_d_STSBL.append(sigma2)
        Mu_xi_d_STSBL.append(mu_xi)
        AICc_d_STSBL.append(AICc(Theta,y,gamma,sigma2))
        if verbose: print('AIC_c='+str(AICc_d_STSBL[-1]))

    q = np.argmin(AICc_d_STSBL)
    gamma_d_STSBL = Gamma_d_STSBL[q]
    sigma2_d_STSBL = Sigma2_d_STSBL[q]
    mu_xi_d_STSBL = Mu_xi_d_STSBL[q]

    return gamma_d_STSBL, sigma2_d_STSBL, mu_xi_d_STSBL