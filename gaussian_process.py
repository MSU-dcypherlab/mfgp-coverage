"""
gaussian_process.py

Implementation of single- and multi-fidelity Gaussian Process learning models capable of hyperparameter
inference and mean/variance prediction.

created by: Paris Perdikaris, Department of Mechanical Engineer, MIT
first created: 5/20/2017
available: https://github.com/paraklas/GPTutorial

modified by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/11/2020
"""

from __future__ import division
from autograd import value_and_grad
import autograd.numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import differential_evolution


class SFGP:
    """
    A single fidelity Gaussian Process class capable of hyperparameter inference and mean/variance prediction
    """

    def __init__(self, X, y, len):
        """
        Initialize the SFGP class.

        :param X: [nxD numpy array] of observation points (D is number of dimensions of input space)
        :param y: [nx1 numpy array] of observation values
        :param len: [scalar] approximate lengthscale of GP (accelerates hyperparameter inference convergence)
        """
        self.D = X.shape[1]
        self.X = X
        self.y = y

        self.hyp = self.init_params(len)

        self.jitter = 1e-8

        self.likelihood(self.hyp)

    def init_params(self, len):
        """
        Initialize hyperparameters of the GP.

        :param len: [scalar] approximate lengthscale of GP (accelerates hyperparameter inference convergence)
        :return: [1xk numpy array] of GP hyperparameters as initialized (k is number of hyperparameters)
        """
        hyp = np.log(np.ones(self.D + 1))
        self.idx_theta = np.arange(hyp.shape[0])
        logsigma_n = np.array([-4.0])
        hyp = np.concatenate([hyp, logsigma_n])

        # manually override starting lengthscale values to accelerate convergence
        hyp[2] = np.log(len)

        return hyp

    def kernel(self, x, xp, hyp):
        """
        Vectorized implementation of a radial basis function kernel.
        
        :param x: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param xp: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [nxn numpy array] of kernel values between all n^2 pairs of points in (x, xp)
        """
        output_scale = np.exp(hyp[1])
        lengthscales = np.exp(hyp[2])
        diffs = np.expand_dims(x / lengthscales, 1) - \
                np.expand_dims(xp / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def likelihood(self, hyp):
        """
        Compute the negative log-marginal likelihood of model given observations (self.X, self.y) and hyperparameters.
        
        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [scalar] negative log-marginal likelihood of model
        """
        X = self.X
        y = self.y

        N = y.shape[0]

        logsigma_n = hyp[-1]
        sigma_n = np.exp(logsigma_n)

        theta = hyp[self.idx_theta]

        K = self.kernel(X, X, theta) + np.eye(N) * sigma_n
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)
        self.L = L

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        NLML = 0.5 * np.matmul(np.transpose(y), alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5 * np.log(2. * np.pi) * N
        return NLML[0, 0]

    def train(self):
        """
        Trains hyperparameters of GP model by minimizing the negative log-marginal likelihood on given data.
        Prints progress of training at each step.
        For best training results, use with 100-300 training points. Kernel computations are O(n^2) leading to fast
        growth, but small training sets may not lead to reliable hyperparameter inference.
        
        :return: None
        """
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
                          method='L-BFGS-B', callback=self.callback)
        self.hyp = result.x

    def predict(self, X_star):
        """
        Return posterior mean and variance conditioned on provided self.X, self.y data
        at a set of test points specified in X_star.

        :param X_star: [nxD numpy array] of test points at which to predict (D is number of dimensions of input space)
        :return: [2-value tuple] of
            [nx1 numpy array] of mean predictions at points in X_star
            [nxn numpy array] of covariance prediction of points in X_star (diagonal is variance at points in X_star)
        """
        X = self.X
        mean = self.hyp[0]
        y = self.y - mean

        L = self.L

        theta = self.hyp[self.idx_theta]

        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        pred_u_star = np.matmul(psi, alpha)
        pred_u_star += mean

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi, beta)

        return pred_u_star, var_u_star

    def ExpectedImprovement(self, X_star):
        """
        Compute the expected improvement by sampling points in X_star.

        :param X_star: [nxD numpy array] of points at which to compute EI (D is number of dimensions of input space)
        :return: [nx1 numpy array] of values of expected improvement at each point in X_star
        """
        X = self.X
        y = self.y

        L = self.L

        theta = self.hyp[self.idx_theta]

        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        pred_u_star = np.matmul(psi, alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi, beta)
        var_u_star = np.abs(np.diag(var_u_star))[:, None]

        # Expected Improvement
        best = np.min(y)
        Z = (best - pred_u_star) / var_u_star
        EI_acq = (best - pred_u_star) * norm.cdf(Z) + var_u_star * norm.pdf(Z)

        return EI_acq

    def draw_prior_samples(self, X_star, N_samples=1):
        """
        Draw N_samples from prior distribution evaluated at points specified in X_star.

        :param X_star: [nxD numpy array] of points at which to compute prior (D is number of dimensions of input space)
        :param N_samples: [scalar] number of points to sample from prior evaluated at X_star
        :return: [nxN_samples numpy array] of prior evaluations at points in X_star
        """
        N = X_star.shape[0]
        theta = self.hyp[self.idx_theta]
        K = self.kernel(X_star, X_star, theta)
        return np.random.multivariate_normal(np.zeros(N), K, N_samples).T

    def draw_posterior_samples(self, X_star, N_samples=1):
        """
        Draw N_samples from posterior distribution evaluated at points specified in X_star.

        :param X_star: [nxD numpy array] of points at which to compute post (D is number of dimensions of input space)
        :param N_samples: [scalar] number of points to sample from posterior evaluated at X_star
        :return: [nxN_samples numpy array] of posterior evaluations at points in X_star
        """
        X = self.X
        y = self.y

        L = self.L

        theta = self.hyp[self.idx_theta]

        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        pred_u_star = np.matmul(psi, alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi, beta)

        return np.random.multivariate_normal(pred_u_star.flatten(),
                                             var_u_star, N_samples).T

    def callback(self, params):
        """
        Callback evaluated in hyperparameter training process. Computes and displays current negative log-marginal
        likelihood of model.

        :param params: [1xk numpy array] of hyperparameters of GP (k is number of hyperparameters)
        :return: None
        """
        print("Log likelihood {}".format(self.likelihood(params)))

    def updt_info(self, X_new, y_new):
        """
        Update model with new X observation points and y observation values. Recompute Cholesky decomposition of
        covariance matrix stored in model.

        :param X_new: [nxD numpy array] of observation points (D is number of dimensions of input space)
        :param y_new: [nx1 numpy array] of observation values
        :return: None
        """
        self.X = X_new
        X = X_new

        self.y = y_new
        y = y_new

        hyp = self.hyp

        N = y.shape[0]

        logsigma_n = hyp[-1]
        sigma_n = np.exp(logsigma_n)

        theta = hyp[self.idx_theta]

        K = self.kernel(X, X, theta) + np.eye(N) * sigma_n
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)
        self.L = L

    def updt(self, X_addition, y_addition):
        """
        Update model with additional X observation points and y observation values. Recompute Cholesky decomposition of
        covariance matrix stored in model.

        :param X_addition: [nxD numpy array] of observation points to be added (D is number of dims of input space)
        :param y_addition: [nx1 numpy array] of observation values to be added
        :return: None
        """
        self.X = np.vstack((self.X, X_addition))
        self.y = np.vstack((self.y, y_addition))
        self.updt_info(self.X, self.y)


class MFGP:
    """
    A multi-fidelity (2-level) Gaussian Process class capable of hyperparameter inference and mean/variance prediction
    """

    def __init__(self, X_L, y_L, X_H, y_H, len_L, len_H):
        """
        Initialize the MFGP class.

        :param X_L: [nxD numpy array] of lofi observation points (D is number of dimensions of input space)
        :param y_L: [nxD numpy array] of lofi observation values
        :param X_H: [nxD numpy array] of hifi observation points (D is number of dimensions of input space)
        :param y_H: [nxD numpy array] of hifi observation values
        :param len_L: [scalar] approximate lofi lengthscale of GP (accelerates hyperparameter inference convergence)
        :param len_H: [scalar] approximate hifi lengthscale of GP (accelerates hyperparameter inference convergence)
        """
        self.D = X_H.shape[1]
        self.X_L = X_L
        self.y_L = y_L
        self.X_H = X_H
        self.y_H = y_H
        self.L = np.empty([0, 0])
        self.idx_theta_L = np.empty([0, 0])
        self.idx_theta_H = np.empty([0, 0])

        self.hyp = self.init_params(len_L, len_H)

        self.jitter = 1e-8

    def init_params(self, len_L, len_H):
        """
        Initialize hyperparameters of the GP.

        :param len_L: [scalar] approximate lofi lengthscale of GP (accelerates hyperparameter inference convergence)
        :param len_H: [scalar] approximate hifi lengthscale of GP (accelerates hyperparameter inference convergence)
        :return: [1xk numpy array] of GP hyperparameters as initialized (k is number of hyperparameters)
        """
        hyp = np.ones(self.D + 1)
        hyp[0] = 0
        self.idx_theta_L = np.arange(hyp.shape[0])

        hyp = np.concatenate((hyp, hyp))
        self.idx_theta_H = np.arange(self.idx_theta_L[-1] + 1, hyp.shape[0])

        rho = np.array([1.0])
        sigma_n = np.array([0.01, 0.01])
        hyp = np.concatenate((hyp, rho, sigma_n))

        # manually override starting lengthscale values to accelerate convergence
        hyp[2] = np.log(len_L)
        hyp[5] = np.log(len_H)

        return hyp

    def kernel(self, x, xp, hyp):
        """
        Vectorized implementation of a radial basis function kernel.

        :param x: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param xp: [nxD numpy array] of points at which to evaluate kernel (D is number of dimensions of input space)
        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [nxn numpy array] of kernel values between all n^2 pairs of points in (x, xp)
        """
        output_scale = np.exp(hyp[1])
        lengthscales = np.exp(hyp[2])
        diffs = np.expand_dims(x / lengthscales, 1) - \
                np.expand_dims(xp / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def likelihood(self, hyp):
        """
        Compute the negative log-marginal likelihood of model given observations (self.X, self.y) and hyperparameters.

        :param hyp: [1xk numpy array] of hyperparameters of GP utilized in computation (k is number of hyperparameters)
        :return: [scalar] negative log-marginal likelihood of model
        """
        rho = np.exp(hyp[-3])
        sigma_n_L = np.exp(hyp[-2])
        sigma_n_H = np.exp(hyp[-1])
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]
        mean_L = theta_L[0]
        mean_H = rho * mean_L + theta_H[0]

        X_L = self.X_L
        y_L = self.y_L
        X_H = self.X_H
        y_H = self.y_H

        y_L = y_L - mean_L
        y_H = y_H - mean_H

        y = np.vstack((y_L, y_H))

        NL = y_L.shape[0]
        NH = y_H.shape[0]
        N = y.shape[0]

        K_LL = self.kernel(X_L, X_L, theta_L) + np.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(X_L, X_H, theta_L)
        K_HH = rho ** 2 * self.kernel(X_H, X_H, theta_L) + \
               self.kernel(X_H, X_H, theta_H) + np.eye(NH) * sigma_n_H
        K = np.vstack((np.hstack((K_LL, K_LH)),
                       np.hstack((K_LH.T, K_HH))))
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)
        self.L = L

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        NLML = 0.5 * np.matmul(np.transpose(y), alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5 * np.log(2. * np.pi) * N
        return NLML[0, 0]

    # Minimizes the negative log-marginal likelihood
    def train(self):
        """
        Trains hyperparameters of GP model by minimizing the negative log-marginal likelihood on given data.
        Prints progress of training at each step.
        For best training results, use with 100-300 training points in each fidelity. Kernel computations are O(n^2)
        leading to fast growth, but small training sets may not lead to reliable hyperparameter inference.

        :return: None
        """
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
                          method='L-BFGS-B', callback=self.callback)
        self.hyp = result.x

    def predict(self, X_star):
        """
        Return posterior mean and variance conditioned on provided self.X, self.y data
        at a set of test points specified in X_star.

        :param X_star: [nxD numpy array] of test points at which to predict (D is number of dimensions of input space)
        :return: [2-value tuple] of
            [nx1 numpy array] of mean predictions at points in X_star
            [nxn numpy array] of covariance prediction of points in X_star (diagonal is variance at points in X_star)
        """
        hyp = self.hyp
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]
        rho = np.exp(hyp[-3])
        mean_L = theta_L[0]
        mean_H = rho * mean_L + theta_H[0]

        X_L = self.X_L
        y_L = self.y_L - mean_L
        X_H = self.X_H
        y_H = self.y_H - mean_H
        L = self.L

        y = np.vstack((y_L, y_H))

        psi1 = rho * self.kernel(X_star, X_L, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, X_H, theta_L) + \
               self.kernel(X_star, X_H, theta_H)
        psi = np.hstack((psi1, psi2))

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        pred_u_star = mean_H + np.matmul(psi, alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                     self.kernel(X_star, X_star, theta_H) - np.matmul(psi, beta)

        return pred_u_star, var_u_star

    def pred_var(self, x, X_L_new, X_H_new):
        """
        Return posterior variance at test points specified in x and newly-sampled points in X_L and X_H.
        Runs faster than predict if only variance is needed. Allows prospective prediction with points not yet in
        conditioning set of model

        :param x: [nxD numpy array] of test points to evaluate at (D is number of dimensions of input space)
        :param X_H_new: [nxD numpy array] of new hifi points to condition on (D is number of dimensions of input space)
        :param X_L_new: [nxD numpy array] of new lofi points to condition on (D is number of dimensions of input space)
        :return: [nxn numpy array] of covariance prediction of points in x (diagonal is variance at points in X_star)
        """
        hyp = self.hyp
        rho = np.exp(hyp[-3])
        sigma_n_L = np.exp(hyp[-2])
        sigma_n_H = np.exp(hyp[-1])
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]

        X_L = np.vstack((self.X_L, X_L_new))
        X_H = np.vstack((self.X_H, X_H_new))

        NL = X_L.shape[0]
        NH = X_H.shape[0]
        N = NL + NH

        K_LL = self.kernel(X_L, X_L, theta_L) + np.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(X_L, X_H, theta_L)
        K_HH = rho ** 2 * self.kernel(X_H, X_H, theta_L) + \
               self.kernel(X_H, X_H, theta_H) + np.eye(NH) * sigma_n_H
        K = np.vstack((np.hstack((K_LL, K_LH)),
                       np.hstack((K_LH.T, K_HH))))
        L = np.linalg.cholesky(K + np.eye(N) * self.jitter)

        psi1 = rho * self.kernel(x, X_L, theta_L)
        psi2 = rho ** 2 * self.kernel(x, X_H, theta_L) + \
               self.kernel(x, X_H, theta_H)
        psi = np.hstack((psi1, psi2))

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L, psi.T))
        var_u_star = rho ** 2 * self.kernel(x, x, theta_L) + \
                     self.kernel(x, x, theta_H) - np.matmul(psi, beta)
        return var_u_star

    def callback(self, params):
        """
        Callback evaluated in hyperparameter training process. Computes and displays current negative log-marginal
        likelihood of model.

        :param params: [1xk numpy array] of hyperparameters of GP (k is number of hyperparameters)
        :return: None
        """
        print("Log likelihood {}".format(self.likelihood(params)))

    def updt_info(self, X_L_new, y_L_new, X_H_new, y_H_new):
        """
        Update model with new X observation points and y observation values. Recompute Cholesky decomposition of
        covariance matrix stored in model.

        :param X_L_new: [nxD numpy array] of lofi observation points (D is number of dimensions of input space)
        :param y_L_new: [nx1 numpy array] of lofi observation values
        :param X_H_new: [nxD numpy array] of hifi observation points (D is number of dimensions of input space)
        :param y_H_new: [nx1 numpy array] of hifi observation values
        :return: None
        """
        self.X_L = X_L_new
        self.y_L = y_L_new
        self.X_H = X_H_new
        self.y_H = y_H_new

        hyp = self.hyp
        rho = np.exp(hyp[-3])
        sigma_n_L = np.exp(hyp[-2])
        sigma_n_H = np.exp(hyp[-1])
        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]

        X_L = self.X_L
        X_H = self.X_H

        NL = X_L.shape[0]
        NH = X_H.shape[0]
        N = NL + NH

        K_LL = self.kernel(X_L, X_L, theta_L) + np.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(X_L, X_H, theta_L)
        K_HH = rho ** 2 * self.kernel(X_H, X_H, theta_L) + \
               self.kernel(X_H, X_H, theta_H) + np.eye(NH) * sigma_n_H
        K = np.vstack((np.hstack((K_LL, K_LH)),
                       np.hstack((K_LH.T, K_HH))))
        self.L = np.linalg.cholesky(K + np.eye(N) * self.jitter)

    def updt_hifi(self, X_H_addition, y_H_addition):
        """
        Update model with additional hifi observation points and values. Recompute Cholesky decomposition of
        covariance matrix stored in model.

        :param X_H_addition: [nxD numpy array] of hifi observation points to be added (D is num of dims of input space)
        :param y_H_addition: [nx1 numpy array] of hifi observation values to be added
        :return: None
        """
        self.X_H = np.vstack((self.X_H, X_H_addition))
        self.y_H = np.vstack((self.y_H, y_H_addition))
        self.updt_info(self.X_L, self.y_L, self.X_H, self.y_H)

    def get_neg_var(self, x, thrd, c, X_L_new, X_H_new):
        """
        Get negative variance at points in X considering new observations in X_L_new and X_H_new

        :param x: [nxD numpy array] of test points to evaluate at (D is number of dimensions of input space)
        :param thrd: [scalar] threshold value
        :param c: [scalar] scaling factor to determine cutoff against threshold
        :param X_L_new: [nxD numpy array] of lofi observation points (D is number of dimensions of input space)
        :param X_H_new: [nxD numpy array] of hifi observation points (D is number of dimensions of input space)
        :return: [nxn numpy array] of negative cov prediction of points in x (diagonal is variance at points in X_star)
        """
        x = x[None, :]
        mean, var_old = self.predict(x)
        if mean + c * var_old <= thrd:
            return 0
        # elif mean - c * var_old >= thrd:
        #     return 0
        else:
            var = self.pred_var(x, X_L_new, X_H_new)
            return -var

    def get_max_var(self, Bounds, Thrd, c, X_L_new, X_H_new):
        """
        Get maximum variance at points inside Bounds considering new observations in X_L_new and X_H_new

        :param Bounds: [2x2 numpy array] of bounds to limit consideration within
        :param Thrd: [scalar] threshold value used in get_neg_var
        :param c: [scalar] scaling factor to determine cutoff against threshold used in get_neg_var
        :param X_L_new: [nxD numpy array] of lofi observation points (D is number of dimensions of input space)
        :param X_H_new: [nxD numpy array] of hifi observation points (D is number of dimensions of input space)
        :return: points of maximum variance within bounds
        """
        Bounds = ([Bounds[0][0], Bounds[1][0]], [Bounds[0][1], Bounds[1][1]])
        result = differential_evolution(self.get_neg_var, Bounds, args=(Thrd, c, X_L_new, X_H_new), init='random')
        return result.x[None, :], -result.fun
