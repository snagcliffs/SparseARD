import numpy as np
import scipy as sp
from sklearn.linear_model import LassoLars as Lasso_sk

def AICc(Theta, y, gamma, sigma2):
    """
    Sample size corrected Akaike information criteria
    """

    m = len(y)
    k = np.count_nonzero(gamma)+1 # +1 since we also fit sigma

    if m-k-1 <= 0: return np.inf

    mu_xi = posterior_mean([Theta,y], gamma, sigma2)
    sample_correction = (2*k**2+2*k)/(m-k-1)

    return 2*k + SBL_cost([Theta, y], sigma2, gamma, mu_xi) + sample_correction

def FiniteDiff(X, dt, order_diff, order_acc = 6, axis = 0):
    """
    order_acc should be in [2,4,6]
    high order_diff near edges will be <= order_acc
    """
    
    if axis == 1: 
        return FiniteDiff(X.T, dt, order_diff, order_acc).T
    if order_diff > 4:
        return FiniteDiff(FiniteDiff(X, dt, 4, order_acc), dt, order_diff-4, order_acc)
    if len(X.shape) == 1:
        return FiniteDiff(X.reshape(len(X),1), dt, order_diff, order_acc).flatten()

    dX = np.zeros_like(X)
    m,n = X.shape
    
    central_templates = {1:{2 : np.array([-1/2,0,1/2]),\
                            4 : np.array([1/12,-2/3,0,2/3,-1/12]),\
                            6 : np.array([-1/60,3/20,-3/4,0,3/4,-3/20,1/60])},\
                         2:{2 : np.array([1,-2,1]),\
                            4 : np.array([-1/12,4/3,-5/2,4/3,-1/12]),\
                            6 : np.array([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90])},\
                         3:{2 : np.array([-1/2,1,0,-1,1/2]),\
                            4 : np.array([1/8,-1,13/8,0,-13/8,1,-1/8]),\
                            6 : np.array([-7/240,3/10,-169/120,61/30,0,-61/30,169/120,-3/10,7/240])},\
                         4:{2 : np.array([1,-4,6,-4,1]),\
                            4 : np.array([-1/6,2,-13/2,28/3,-13/2,2,-1/6]),\
                            6 : np.array([7/240,-2/5,169/60,-122/15,91/8,-122/15,169/60,-2/5,7/240])}}
    
    forward_templates = {1:{2 : np.array([-3/2,2,-1/2]),\
                            4 : np.array([-25/12,4,-3,4/3,-1/4]),\
                            6 : np.array([-49/20,6,-15/2,20/3,-15/4,6/5,-1/6])},\
                         2:{2 : np.array([2,-5,4,-1]),\
                            4 : np.array([15/4,-77/6,107/6,-13,61/12,-5/6]),\
                            6 : np.array([469/90,-223/10,879/20,-949/18,41,-201/10,1019/180,-7/10])},\
                         3:{2 : np.array([-5/2,9,-12,7,-3/2]),\
                            4 : np.array([-49/8,29,-461/8,62,-307/8,13,-15/8]),\
                            6 : np.array([-801/80,349/6,-18353/120,2391/10,-1457/6,4891/30,-561/8,527/30,-469/240])},\
                         4:{2 : np.array([3,-14,26,-24,11,-2]),\
                            4 : np.array([28/3,-111/2,142,-1219/6,176,-185/2,82/3,-7/2]),\
                            6 : np.array([1069/80,-1316/15,15289/60,-2144/5,10993/24,-4772/15,2803/20,-536/15,967/240,0])}} # this is actually order_acc = 5
    
    w_central = int((len(central_templates[order_diff][order_acc])-1)/2)
    w_forward = len(forward_templates[order_diff][order_acc])

    for i in range(n):
        for j in range(w_central, m-w_central):
            dX[j,i] = np.inner(X[j-w_central:j+w_central+1,i],central_templates[order_diff][order_acc])
            
        for j in range(w_central):
            dX[j,i] = np.inner(X[j:j+w_forward,i],forward_templates[order_diff][order_acc])
            dX[m-j-1,i] = (-1)**order_diff*np.inner(X[m-j-w_forward:m-j,i],forward_templates[order_diff][order_acc][::-1])

    dX = dX / dt**order_diff    
    return dX

def deg_p_polynomials(n,deg):
    """
    Given n and deg, return all coefficient lists of monomials in n variables of degree deg.
    Example: deg_p_polynomials(2,3) returns [[3,0],[2,1],[1,2],[0,3]]
    """
    
    if deg == 0: return [[0 for _ in range(n)]]
    if n == 1: return [[deg]]

    polys = []    
    for j in range(deg+1):
        smaller = deg_p_polynomials(n-1,deg-j)
        for small_poly in smaller:
            polys.append([j]+small_poly)
    return polys

def polynomials(n,max_deg):
    """
    Given n and max_deg, return all coefficient lists of monomials in n variables of degree <= max_deg.
    Example: deg_p_polynomials(2,3) returns [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2],[3,0],[2,1],[1,2],[0,3]]
    """
    
    polys = []
    for p in range(0,max_deg+1):
        polys = polys + deg_p_polynomials(n,p)
        
    return polys

def polynomial_feature_maps(n, max_deg):
    
    powers = polynomials(n,max_deg)
    def f(x,y): return np.prod(np.power(x, y), axis = 1)
    
    polynomial_features = [lambda x, y = np.array(power): f(x,y).reshape(x.shape[0],1) \
                               for power in powers]
    
    descriptions = [np.array(power) for power in powers]
        
    return polynomial_features, descriptions

def Lasso(Theta, y, c, xi = np.array([0]), tol = 1e-10, maxit = 2500, verbose=False):
    """
    Uses coordinate descent to solve weighted Lasso problem
    argmin (1/2)*||Theta xi-Y||_2^2 + <c, |xi|>
    """
    
    m,d = Theta.shape
    y = y.reshape(m,1)
    
    # Initialize xi if none is given
    if xi.size != d:
        xi = np.linalg.lstsq(Theta,y,rcond=None)[0]
    xi_0 = np.copy(xi)
    N = np.linalg.norm(Theta, axis = 0)**2
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
        
        xi_0[:] = xi[:]
        residual = y - Theta.dot(xi)
        
        # Generate a random permutation of coefficients to pass over
        for j in np.random.permutation(d):
            
            sub_residual = residual + Theta[:,j].reshape(m,1)*xi[j]            
            xi[j] = soft_threshold(Theta[:,j].reshape(1,m).dot(sub_residual)/N[j], c[j]/N[j])
            residual = residual + Theta[:,j].reshape(m,1)*(xi_0[j]-xi[j])

        # Break condition
        if np.linalg.norm(xi-xi_0, np.inf) < tol: 
            if verbose: print('Lasso converged in', iters+1, 'iterations.')
            return xi
    
    if verbose: print('Lasso did not converge in', maxit, 'iterations.')
    return xi

def Lasso_FISTA(Theta, y, c, xi = np.array([0]), tol = 1e-10, maxit = 2500, verbose=False, return_cost = False):
    """
    Uses accelerated proximal gradient (FISTA) to solve weights Lasso problem
    argmin (1/2)*||Theta xi-Y||_2^2 + <c, |xi|>
    """
    
    m,d = Theta.shape
    y = y.reshape(m,1)
    
    # Initialize xi if none is given
    if xi.size != d:
        xi = np.linalg.lstsq(Theta,y)[0]
    xi_0 = np.copy(xi)
        
    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(Theta.T.dot(Theta),2)
    count = 0
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = xi + iters/float(iters+1)*(xi - xi_0)
        xi_0[:] = xi[:]
        z = z - Theta.T.dot(Theta.dot(z)-y)/L
        for j in range(d): xi[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-c[j]/L,0]))

        # Break condition
        # Note FISTA method does not monotonically decrease Lasso objective
        if np.linalg.norm(xi-xi_0) < tol:
            count = count+1
            if count == 5: 
                if verbose: print('Lasso converged in', iters+1, 'iterations.')
                return xi
        else: count = 0
    
    if verbose: print('Lasso did not converge in', maxit, 'iterations.')
    return xi

def soft_threshold(x, lam):
    if np.abs(x) < lam: return 0
    else: return x - np.sign(x)*lam
    
def SBL_cost(dataset, sigma2, gamma, xi):

        sq_err = np.linalg.norm(dataset[1] - dataset[0].dot(xi))**2/sigma2
        ridge_penalty = np.sum([xi[i]**2/gamma[i] for i in range(xi.size) if gamma[i] > 0])
        log_det_penalty = np.linalg.slogdet(sigma2*np.eye(dataset[1].size) + \
                dataset[0].dot(np.diag(gamma.flatten())).dot(dataset[0].T))[1]

        return sq_err+ridge_penalty+log_det_penalty
    
def posterior_mean(dataset, gamma, sigma2):
    
    Theta, y = dataset
    m,n = Theta.shape
        
    xi_mean = np.zeros((n,1))
    nnz = np.where(gamma > 0)[0]

    xi_mean[nnz] = np.linalg.solve(Theta[:,nnz].T.dot(Theta[:,nnz]) + \
            sigma2*np.diag(gamma[nnz,0]**-1), 
            Theta[:,nnz].T.dot(y))

    return xi_mean

def posterior_precision(dataset, gamma, sigma2):
    """
    Only defined where gamma != 0
    """
    
    Theta, y = dataset    
    nnz = np.where(gamma > 0)[0]
    return Theta[:,nnz].T.dot(Theta[:,nnz])/sigma2 + np.diag(gamma[nnz,0]**-1)

def posterior_covariance(dataset, gamma, sigma2):

    Theta, y = dataset
    m,n = Theta.shape
    
    posterior_cov = np.zeros((n,n))
    nnz = np.where(gamma > 0)[0]
    nnz_cov = np.linalg.inv(posterior_precision(dataset, gamma, sigma2))

    for i in range(len(nnz)):
        for j in range(len(nnz)):
            posterior_cov[nnz[i],nnz[j]] = nnz_cov[i,j]

    return posterior_cov
    
def sparse_estimate_sigma(Theta, y, s_hat):

    m,n = Theta.shape

    s_hat = np.min([n/2, m/2, 50])
        
    c = np.linalg.norm(Theta.T.dot(y), np.inf)*np.ones(n)

    for j in range(500):

        xi_lasso = Lasso(Theta,y,c)

        if np.count_nonzero(xi_lasso) > s_hat: 
            break
        
        c = 0.95*c
    
    G = np.where(xi_lasso != 0)[0]
    xi_sparse = np.linalg.lstsq(Theta[:,G], y,rcond=None)[0]

    sigma = np.std(y - Theta[:,G].dot(xi_sparse))

    return sigma

def SBL(dataset, gamma = None, alpha = 1, lam = 0, eta = np.inf, sigma2=None, maxit = 500, tol = 1e-8, \
    verbose = True, estimate_sigma = False, regularize = False):
    """
    Solves SBL with penalty
    """

    Theta, y = dataset
    m,n = Theta.shape

    xi_solver = Lasso_sk(alpha=1/m, \
                         fit_intercept=False, \
                         max_iter=10000)

    xi = np.zeros((n,1))
    if gamma is None: gamma = np.zeros((n,1))
    gamma_old = np.copy(gamma)

    if sigma2 == None: 
        
        # Initialize sigma2 as variance of Lasso error for suitably sparse lasso
        # s_hat is guess for s, the number of nonzero terms in xi
        # if s_hat < s then initial sigma2 will be >> true variance
        # if s_hat > s then initial sigma2 can still be close
        s_hat = np.min([n/2, m/2, 50])            
        sigma2 = 100*sparse_estimate_sigma(Theta, y, s_hat)**2
        estimate_sigma = True

        if verbose: print('Initial estimate of sigma^2:', sigma2)
        
    if regularize: penalty = np.sum(lam*gamma/sigma2*(gamma < eta).astype(int) + lam*(gamma >= eta).astype(int))
    else: penalty = 0
        
    SBL_loss = [SBL_cost(dataset, alpha*sigma2, gamma, xi)+penalty]

    for iter in range(maxit):

        # compute c
        Sigma_y = alpha*sigma2*np.eye(m) + Theta.dot(np.diag(gamma.flatten())).dot(Theta.T)
        
        Sigma_y_inv_Theta = np.linalg.solve(Sigma_y, Theta)
        c = np.array([np.inner(Theta[:,j].T, Sigma_y_inv_Theta[:,j]) for j in range(n)])
        
        if regularize: 
            penalty = np.sum(lam*gamma/sigma2*(gamma < eta).astype(int) + lam/sigma2*(gamma >= eta).astype(int))
            grad_penalty = lam/sigma2*(gamma < eta).astype(int)
            c = np.sqrt(c + grad_penalty.flatten())
        else: 
            penalty = 0
            c = np.sqrt(c)
            
        # solve weighted lasso problem
        #
        #   xi = argmin (1/2)||Theta xi - y||^2 + sigma^2 <c,|xi|>
        #        
        xi = np.divide(xi_solver.fit(Theta.dot(np.diag(c**-1))/(alpha*sigma2), y).coef_.reshape(n,1), \
                       c.reshape(n,1))/(alpha*sigma2)
        
        if verbose: print('Lasso converged in', xi_solver.n_iter_, 'iterations.')

        # update gamma
        gamma = np.abs(xi)*(c.reshape(*xi.shape))**-1

        # evaluate break condition
        SBL_loss.append(SBL_cost(dataset, alpha*sigma2, gamma, xi)+penalty)

        if iter > 0 and np.linalg.norm(gamma-gamma_old, np.inf) <= tol: 
            if verbose: print('SBL algorithm converged in', iter, 'iterations.')
            break

        gamma_old[:] = gamma[:]

        if iter == maxit-1: print('SBL algorithm failed to converge to specified tolerance in', maxit, 'iterations.')
            
        # update noise covariance if needed
        if estimate_sigma:

            rss = np.linalg.norm(dataset[1]-np.dot(dataset[0],posterior_mean(dataset, gamma, sigma2)))**2
            
            if m>n:
                Sigma_xi = posterior_covariance(dataset, gamma, sigma2)
                denom = m-n+np.sum([Sigma_xi[j,j]/gamma[j] for j in range(n) if gamma[j]!=0])
                sigma2 = rss / denom

            # Grid search
            else:   
                Sigma2 = np.logspace(np.log(np.var(dataset[1]))-10,np.log(np.var(dataset[1])),101)
                SBL_cost_Sigma2 = np.array([SBL_cost([Theta, y], sigma2, gamma, xi) \
                                            for sigma2 in Sigma2])
                sigma2 = Sigma2[np.argmin(SBL_cost_Sigma2)]
    
    if lam != 0 and regularize == False: 
        
        # Solve regularized problem with unregularized solution as IC
        gamma_reg, sigma2_reg, xi_reg, SBL_loss_reg = SBL(dataset, gamma=gamma, lam=lam, eta=eta, sigma2=sigma2, \
                   maxit = maxit, tol = tol, verbose = verbose, estimate_sigma=estimate_sigma, regularize = True, alpha=alpha)
        
        return [gamma, gamma_reg],[sigma2, sigma2_reg], [xi, xi_reg], [SBL_loss, SBL_loss_reg]
    
    else: return gamma, sigma2, xi, SBL_loss

def M_STSBL(A, y, tau, sigma2 = None, verbose = True, maxit = 500, estimate_sigma = True, tol=1e-8, warm_start = None):
    """
    Magnitude based Sequential Threshold Sparse Bayesian Learning algorithm for finding sparse 
    approximation to A^{-1}y.
    """
    
    m,n = A.shape

    if warm_start is None:
        gamma, sigma2, x, _ = SBL([A,y], \
                                        sigma2=sigma2, \
                                        estimate_sigma=estimate_sigma, \
                                        verbose=verbose, \
                                        maxit=maxit, \
                                        tol=tol)  
    else:
        gamma, sigma2, x = np.copy(warm_start)

    G = np.where(np.abs(x) > tau)[0]
    Gc = [j for j in range(n) if j not in G]
    
    if len(Gc) == 0: return x, gamma, sigma2
    else:
        x[Gc] = 0
        gamma[Gc] = 0
        if len(G)>0: x[G], gamma[G], sigma2 = M_STSBL(A[:,G], y, tau, sigma2, verbose, maxit, estimate_sigma, tol)    
        return x, gamma, sigma2

def L_STSBL(A, y, tau, sigma2 = None, verbose = True, maxit = 500, \
                      estimate_sigma = True, tol=1e-8, warm_start = None):
    """
    Likelihood Sequential Threshold Sparse Bayesian Learning algorithm for finding sparse 
    approximation to A^{-1}y.

    This is L-STSBL in paper
    """
    
    m,n = A.shape

    if warm_start is None:
        gamma, sigma2, x, _ = SBL([A,y], \
                                        sigma2=sigma2, \
                                        estimate_sigma=estimate_sigma, \
                                        verbose=verbose, \
                                        maxit=maxit, \
                                        tol=tol)  
    else:
        gamma, sigma2, x = np.copy(warm_start)
        
    likelihood_0 = np.zeros(n)    
    marginal_posterior_cov = np.diag(posterior_covariance([A,y], gamma, sigma2)) 
    
    for j in range(n):
        if marginal_posterior_cov[j] == 0: likelihood_0[j] = np.inf
        else: likelihood_0[j] = np.exp(-0.5*x[j]**2/marginal_posterior_cov[j])/np.sqrt(2*np.pi*marginal_posterior_cov[j])
        
    G = np.where(np.abs(likelihood_0) < tau)[0]
    Gc = [j for j in range(n) if j not in G]
    
    if len(Gc) == 0: return x, gamma, sigma2
    else:
        x[Gc] = 0
        gamma[Gc] = 0
        if len(G)>0: x[G], gamma[G], sigma2 = L_STSBL(A[:,G], y, tau, sigma2, verbose, maxit, estimate_sigma, tol)    
        return x, gamma, sigma2

def MAP_STSBL(A, y, tau, sigma2 = None, verbose = True, maxit = 500, \
           estimate_sigma = True, method = 'diagonal', tol=1e-8, warm_start=None):
    """
    MAP Sequential Threshold Sparse Bayesian Learning algorithm for finding sparse 
    approximation to A^{-1}y.

    This is MAP-STSBL
    """
    
    m,n = A.shape

    if warm_start is None:
        gamma, sigma2, x, _ = SBL([A,y], \
                                sigma2=sigma2, \
                                estimate_sigma=estimate_sigma, \
                                verbose=verbose, \
                                maxit=maxit, \
                                tol=tol)  
    else:
        gamma, sigma2, x = np.copy(warm_start) 

    neg_log_likelihood_0 = np.zeros(n)   
    
    Sigma_x = posterior_covariance([A,y], gamma, sigma2)
    Prec_x = posterior_precision([A,y], gamma, sigma2)
    nnz = np.where(gamma > 0)[0]

    if method == 'diagonal':
        for i in range(len(nnz)):
            j = nnz[i]
            neg_log_likelihood_0[j] = 0.5*x[j]**2*Prec_x[i,i]
        Gc = np.where(neg_log_likelihood_0 < tau)[0]

    else:
        Gc = [nnz[i] for i in GreedySPT(Prec_x, x[nnz], tau)]

    G = [j for j in range(n) if j not in Gc]
    
    if len(Gc) == 0: return x, gamma, sigma2
    else:
        x[Gc] = 0
        gamma[Gc] = 0
        if len(G)>0: x[G],gamma[G],sigma2=MAP_STSBL(A[:,G],y,tau,sigma2,verbose,maxit,estimate_sigma,tol)    
        return x, gamma, sigma2


def SPT_criteria(Prec, x, Gc, tau):    
    return 0.5*x[Gc].T.dot(Prec[Gc,:][:,Gc]).dot(x[Gc]) - tau*len(Gc)

def GreedySPT(Prec, x, tau, bwd_freq = np.inf, maxit = None):
    """
    This is an attempt at a greedy approach to the subset selection problem in MAP STSBL
    It is not included in the paper.
    """

    d = x.shape[0]

    if maxit == None: maxit = d + int(d/bwd_freq)
    Gc = set()

    for iter in range(maxit):
        
        S = []

        # Find minimal increase in SPT_criteria
        for j in range(d):
            if j in Gc: S.append(np.inf)
            else: S.append(SPT_criteria(Prec, x, list(Gc.union({j})), tau))

        p = np.argmin(S)

        # Stopping criteria
        if S[p] > 0: break
        else: Gc = Gc.union({p}) 

        if iter % bwd_freq == 0:

            for bwd_iter in range(len(Gc)):
                S = [SPT_criteria(Prec, x, list(Gc.difference({j})), tau) for j in list(Gc)]
                p = np.argmin(S)
                if S[p] > 0: 
                    Gc = Gc.difference({list(Gc)[p]}) 
                    break
    return Gc

def sparsity_err(x_pred,x_true):
    """

    """

    xs = (x_pred == 0).astype(int)
    ys = (x_true == 0).astype(int)

    total = np.sum((xs+ys)%2)
    added = np.count_nonzero((1-xs)*ys) #x_pred != 0 and x_true = 0
    missed = np.count_nonzero(xs*(1-ys))
    
    return total, added, missed

def print_results(x_true, X_hat, labels, metrics = None, print_precision = 5):
    
    max_len_label = np.max([len(label) for label in labels])
    label_spaces = [" "*int(max_len_label-len(label)) for label in labels]
    float_placeholder = '%.'+str(print_precision)+'f'
    
    if metrics == None:
        metrics = ['L1', 'L2', 'sparsity']
                
    if 'L1' in metrics:        
        for x_hat, label, space in zip(X_hat, labels, label_spaces):
            print(label, "l1 err:", space, float_placeholder % np.linalg.norm(x_hat - x_true,1))
        print(' ')

    if 'L2' in metrics:
        for x_hat, label, space in zip(X_hat, labels, label_spaces):
            print(label, "l2 err:", space, float_placeholder % np.linalg.norm(x_hat - x_true,2))
        print(' ')
            
    if 'sparsity' in metrics:
            
        for x_hat, label, space in zip(X_hat, labels, label_spaces):
            print(label, "sparsity err:", space, sparsity_err(x_hat, x_true)[0])
        print(' ')
        
        for x_hat, label, space in zip(X_hat, labels, label_spaces):
            added, missed = sparsity_err(x_hat, x_true)[1:]
            print(label, "missed", missed, "and added", added)

def build_linear_system(u, dt, dx, D = 3, P = 3):
    """
    Constructs a linear system to use in later regression for finding PDE.  

    Input:
        Required:
            u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
        Optional:
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
    Output:
        ut = column vector of length u.size
        Theta = matrix with ((D+1)*(P+1)) of column, each as large as ut
        rhs_description = description of what each column in R is
    """

    n, m = u.shape

    ########################
    # Time derivaitve for the left hand side of the equation
    ########################
    ut = np.reshape(FiniteDiff(u, dt, 1, axis = 1), (n*m,1), order='F')

    ########################
    # Build Theta and description of each column
    ########################
    Theta = np.zeros((n*m, (D+1)*(P+1)))
    ux = np.zeros((n,m))
    rhs_description = ['' for i in range((D+1)*(P+1))]
        
    for d in range(D+1):

        if d > 0:
            ux = FiniteDiff(u, dx, d, axis = 0)

        else: ux = np.ones((n,m)) 
            
        for p in range(P+1):
            Theta[:, d*(P+1)+p] = np.reshape(np.multiply(ux, np.power(u,p)), (n*m), order='F')

            if p == 1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u'
            elif p>1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u^' + str(p)
            if d > 0: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+\
                                                   'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    return ut, Theta, rhs_description







