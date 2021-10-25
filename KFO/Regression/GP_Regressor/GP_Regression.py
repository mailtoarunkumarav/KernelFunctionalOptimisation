import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import datetime
import sys
from Functions import FunctionHelper
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import scipy as sp

import os
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH

import math
from sklearn.model_selection import train_test_split
import pandas as pd
# kernel_type = 0
# number_of_test_datapoints = 20
np.random.seed(500)
# noise = 0.0


class GaussianProcess:

    # Constructor
    def __init__(self, output_gen_time, kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, signal_variance, number_of_dimensions,
                 number_of_observed_samples, X, y, hyper_params_estimation, number_of_restarts_likelihood, lengthscale_bounds,
                 signal_variance_bounds, fun_helper_obj, Xmin, Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights, len_weights_bounds,
                 weight_params_estimation, degree_estimation, degree):

        self.output_gen_time = output_gen_time
        self.kernel_type = kernel_type
        self.number_of_test_datapoints = number_of_test_datapoints
        self.noise = noise
        self.linspacexmin = linspacexmin
        self.linspacexmax = linspacexmax
        self.linspaceymin = linspaceymin
        self.linspaceymax = linspaceymax
        self.signal_variance = signal_variance
        self.number_of_dimensions = number_of_dimensions
        self.number_of_observed_samples = number_of_observed_samples
        self.X = X
        self.y = y
        self.hyper_params_estimation = hyper_params_estimation
        self.number_of_restarts_likelihood = number_of_restarts_likelihood
        self.lengthscale_bounds = lengthscale_bounds
        self.signal_variance_bounds = signal_variance_bounds
        self.fun_helper_obj = fun_helper_obj
        self.xmin = Xmin
        self.Xmax = Xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Xs = Xs
        self.ys = ys
        self.char_len_scale = char_len_scale
        self.len_weights = len_weights
        self.len_weights_bounds = len_weights_bounds
        self.weight_params_estimation = weight_params_estimation
        self.hyper_gp_obj = None
        self.degree_estimation = degree_estimation
        self.degree = degree
        # np.random.seed(random_seed)

    # Define Plot Prior
    def plot_graph(self, plot_params):
        plt.figure(plot_params['plotnum'])
        plt.clf()
        for eachplot in plot_params['plotvalues']:
            if (len(eachplot) == 2):
                plt.plot(eachplot[0], eachplot[1])
            elif (len(eachplot) == 3):
                plt.plot(eachplot[0], eachplot[1], eachplot[2])
            elif (len(eachplot) == 4):
                if(eachplot[3].startswith("label=")):
                    plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:])
                    plt.legend(loc='upper right',prop={'size': 6})
                else:
                    flag = eachplot[3]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

            elif (len(eachplot) == 5):

                if(eachplot[3].startswith("label=")):
                    flag = eachplot[4]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:], lw=eachplot[4][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:], ms=eachplot[4][2:])
                    plt.legend(loc='upper right',prop={'size': 6})

                else:
                    flag = eachplot[3]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

        if 'gca_fill' in plot_params.keys():
            if len(plot_params['gca_fill']) == 3:
                plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                       plot_params['gca_fill'][2],
                                       color="#ddddee")
            else:
                if plot_params['gca_fill'][3].startswith('color'):
                    color = plot_params['gca_fill'][3][6:]
                    plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                           plot_params['gca_fill'][2], color=color)

        plt.axis(plot_params['axis'])
        plt.title(plot_params['title'])
        plt.xlabel(plot_params['xlabel'])
        plt.ylabel(plot_params['ylabel'])
        plt.savefig(plot_params['file'], bbox_inches='tight')


    # Define the kernel function
    def computekernel(self, data_point1, data_point2):

        if self.kernel_type == 'SE':
             result = self.sq_exp_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'MATERN5':
            result = self.matern5_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'MATERN3':
            result = self.matern3_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'MKL':
            result = self.multi_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'POLY':
            result = self.poly_kernel(data_point1, data_point2, self.degree)

        elif self.kernel_type == 'LIN':
            result = self.linear_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'PER':
            result = self.periodic_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        return result

    def matern3_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                each_kernel_val = (signal_variance ** 2) * (1 + (np.sqrt(3)*l2_difference/char_len_scale)) * \
                                  (np.exp((-1 * np.sqrt(3) / char_len_scale) * l2_difference))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def matern5_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                each_kernel_val = (signal_variance**2)* (1 + (np.sqrt(5)*l2_difference/char_len_scale) + (5*(l2_difference**2)/(
                        3*(char_len_scale**2)))) * (np.exp((-1 * np.sqrt(5) * l2_difference / char_len_scale)))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat


    def periodic_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        p = 2
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                each_kernel_val = (signal_variance ** 2) * (np.exp((-2 / (char_len_scale**2)) * ((np.sin(difference*(np.pi/p)))**2)))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def multi_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                l2_difference_sq = np.dot(difference, difference.T)
                sek = (signal_variance ** 2) * (np.exp((-1 / (2*char_len_scale**2)) * l2_difference_sq))
                mat3 = (signal_variance ** 2) * (1 + (np.sqrt(3)*l2_difference/char_len_scale)) * \
                                  (np.exp((-1 * np.sqrt(3) / char_len_scale) * l2_difference))
                lin = signal_variance + np.dot(data_point1[i, :], data_point2[j, :].T) * (char_len_scale**2)
                p = 2
                periodic = (signal_variance ** 2) * (np.exp((-2 / (char_len_scale**2)) * ((np.pi/p)*((np.sin(difference)))**2)))
                each_kernel_val = self.len_weights[0] * sek + self.len_weights[1] * mat3 + self.len_weights[2] * lin + \
                                  self.len_weights[3] * periodic
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat


    def sq_exp_kernel_vanilla(self, data_point1, data_point2, char_length_scale, signal_variance):

        # Define the SE kernel function
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = (signal_variance **2) * np.exp(-(total_squared_distances * (1 / ((char_length_scale**2) * 2.0))))
        return kernel_val

    def sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference_sq = np.dot(difference, difference.T)
                each_kernel_val = (signal_variance ** 2) * (np.exp((-1 / (2*char_len_scale**2)) * l2_difference_sq))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def ard_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        # Element wise squaring the vector of given length scales
        char_len_scale = np.array(char_len_scale) ** 2
        sq_dia_len = np.diag(char_len_scale)
        inv_sq_dia_len = 1/sq_dia_len
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = ((data_point1[i, :] - data_point2[j, :]))
                product1 = np.dot(difference, inv_sq_dia_len)
                final_product = np.dot(product1, difference.T)
                each_kernel_val = (signal_variance**2) * (np.exp((-1 / 2.0) * final_product))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def poly_kernel(self, data_point1, data_point2, degree):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                each_kernel_val = 1+np.power(np.dot(data_point1[i, :], data_point2[j, :].T), degree)
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def linear_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                each_kernel_val = signal_variance + np.dot(data_point1[i, :], data_point2[j, :].T) * (char_len_scale**2)
                # each_kernel_val = each_kernel_val/number_of_observed_samples
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat


    def optimize_log_marginal_likelihood_l(self, input):

        # 0 to n-1 elements represent the nth element
        init_charac_length_scale = np.array(input[: self.number_of_dimensions])
        signal_variance = input[len(input)-1]

        if self.kernel_type == 'SE':
            K_x_x = self.sq_exp_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'MATERN3':
            K_x_x = self.matern3_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'MATERN5':
            K_x_x = self.matern5_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'LIN':
            K_x_x = self.linear_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'PER':
            K_x_x = self.periodic_kernel(self.X, self.X, init_charac_length_scale, signal_variance)

        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood


    def optimize_log_marginal_likelihood_degree(self, input):

        self.degree = degree = input
        K_x_x = self.poly_kernel(self.X, self.X, degree)
        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood


    def optimize_log_marginal_likelihood_weight_params(self, input):

        self.len_weights = input[0:4]
        # Following parameters not used in any computations
        signal_variance = input[len(input) - 1]
        self.signal_variance = signal_variance

        K_x_x = self.multi_kernel(self.X, self.X, self.char_len_scale, signal_variance)
        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood

    def plot_posterior_distribution(self, kernel_type, observations_kernel, optional_hypergp_obj, msg):

        K_xs_xs = self.compute_kernel_matrix_hyperkernel(self.Xs, self.Xs, observations_kernel)

        # K_x_x = self.computekernel(self.X, self.X)
        K_x_x = self.compute_kernel_matrix_hyperkernel(self.X, self.X, observations_kernel)

        # # #NIPS Post processing of the gram matrix
        kernel_mat_eigen_values, kernel_mat_eigen_vectors = np.linalg.eigh(K_x_x)

        eig_sign = np.sign(kernel_mat_eigen_values)
        updated_eigen_diag = np.diag(eig_sign)

        # Spectrum Clip
        # kernel_mat_eigen_values[kernel_mat_eigen_values<0]=0
        # updated_eigen_diag = np.diag(kernel_mat_eigen_values)

        # Clip Indicator Function
        # kernel_mat_eigen_values[kernel_mat_eigen_values < 0] = 0
        # kernel_mat_eigen_values[kernel_mat_eigen_values > 0] = 1
        # updated_eigen_diag = np.diag(kernel_mat_eigen_values)

        K_x_x = np.dot(np.dot(np.dot(kernel_mat_eigen_vectors, updated_eigen_diag), kernel_mat_eigen_vectors.T), K_x_x)

        eye = 1e-10 * np.eye(len(self.X))
        L_x_x = np.linalg.cholesky(K_x_x + eye)

        # K_x_xs = self.computekernel(self.X, self.Xs)
        K_x_xs = self.compute_kernel_matrix_hyperkernel(self.X, self.Xs, observations_kernel)

        # NeurIPS commented
        K_x_xs = np.dot(np.dot(np.dot(kernel_mat_eigen_vectors, updated_eigen_diag), kernel_mat_eigen_vectors.T), K_x_xs)


        factor1 = np.linalg.solve(L_x_x, K_x_xs)
        factor2 = np.linalg.solve(L_x_x, self.y)
        mean = np.dot(factor1.T, factor2).flatten()

        variance = K_xs_xs - np.dot(factor1.T, factor1)
        diag_variance = np.diag(variance)
        standard_deviation = np.sqrt(diag_variance)

        count = 10
        plot_posterior_distr_params = {'plotnum': 'Fig__'+self.output_gen_time+"__"+msg,
                                       # 'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                                       # 'axis': [0, 1, self.linspaceymin, self.linspaceymax],
                                       'axis': [0, 1, 0, 1],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms20'], [self.Xs, self.ys, 'b-', 'label=True Fn'],
                                                      [self.Xs, mean, 'g--', 'label=Mean Fn', 'lw2']],
                                       'title': 'GP Posterior Distribution',
                                       'file': 'GP_Posterior_Distr' + str(count),
                                       'gca_fill': [self.Xs.flat, mean - 2 * standard_deviation,
                                                    mean + 2 * standard_deviation],
                                       'xlabel': 'x',
                                       'ylabel': 'output, f(x)'
                                       }

        PH.printme(PH.p1, "\n\n\n\nX: \n", self.X,
              "y: \n",self.y,
              "\n\nmean: \n", mean,
               "\n\nstd_dev: \n", standard_deviation,
              )

        self.plot_graph(plot_posterior_distr_params)

    def compute_log_marginal_likelihood_hyperkernel(self, observations_kernel):

        K_x_x = self.compute_kernel_matrix_hyperkernel(self.X, self.X, observations_kernel)

        kernel_mat_eigen_values, kernel_mat_eigen_vectors = np.linalg.eigh(K_x_x)

        # # Method 2 - sgn(\lambda_i)
        # # Spectrum Flip
        eig_sign = np.sign(kernel_mat_eigen_values)
        updated_eigen_diag = np.diag(eig_sign)

        # Spectrum Clip
        # kernel_mat_eigen_values[kernel_mat_eigen_values<0]=0
        # updated_eigen_diag = np.diag(kernel_mat_eigen_values)

        # Clip Indicator Function
        # kernel_mat_eigen_values[kernel_mat_eigen_values < 0] = 0
        # kernel_mat_eigen_values[kernel_mat_eigen_values > 0] = 1
        # updated_eigen_diag = np.diag(kernel_mat_eigen_values)

        K_x_x = np.dot(np.dot(np.dot(kernel_mat_eigen_vectors, updated_eigen_diag), kernel_mat_eigen_vectors.T), K_x_x)

        eye = 1e-12 * np.eye(len(self.X))
        Knoise = K_x_x + eye

        if math.isnan(np.linalg.cond(Knoise)):
            PH.printme(PH.p4, "nan value encountered")

        try:
            L_x_x = np.linalg.cholesky(Knoise)
            factor = np.linalg.solve(L_x_x, self.y)

        except np.linalg.LinAlgError:
            PH.printme(PH.p4, "!!!!!!!!!!!Matrix is not positive definite, \nEigen", )
            exit(0)

        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood

    def compute_kernel_matrix_hyperkernel(self, data_point1, data_point2, observations_kernel):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                each_kernel_val = self.hyper_gp_obj.estimate_kernel_for_Xtil(data_point1[i, :], data_point2[j, :], observations_kernel)
                kernel_mat[i, j] = each_kernel_val

        return kernel_mat


    def plot_kernel(self, msg, observations_kernel):

        PH.printme(PH.p1, "plotting kernel for linear kernel\n")
        kernel_mat = np.zeros(shape=(100, 100))
        xbound = np.linspace(0, 2, 100).reshape(-1, 1)
        X1, X2 = np.meshgrid(xbound, xbound)
        for xb_i in range(len(xbound)):
            for xb_j in range(len(xbound)):
                if(self.kernel_type =='HYPER'):
                    if(xb_i == 99 and xb_j == 99):
                        PH.printme(PH.p1, "here")
                    kernel_mat[xb_i][xb_j] = self.hyper_gp_obj.estimate_kernel_for_Xtil(np.array([xbound[xb_i]]), np.array([xbound[xb_j]]),
                                                                                      observations_kernel)
                else:
                    kernel_mat[xb_i][xb_j] = self.computekernel(np.array([xbound[xb_i]]), np.array([xbound[xb_j]]))

        fig = plt.figure(msg)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X1, X2, kernel_mat, rstride=1, cstride=1,
                               cmap='viridis', linewidth=1, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=20)

    def compute_mean_var(self, Xs, X, y):

        # Apply the kernel function to our training points
        K_x_x = self.computekernel(X, X)

        eye = 1e-10 * np.eye(len(X))
        L_x_x = np.linalg.cholesky(K_x_x + eye)

        K_x_xs = self.computekernel(X, Xs)
        factor1 = np.linalg.solve(L_x_x, K_x_xs)
        factor2 = np.linalg.solve(L_x_x, y)
        mean = np.dot(factor1.T, factor2).flatten()

        K_xs_xs = self.computekernel(Xs, Xs)
        variance = K_xs_xs - np.dot(factor1.T, factor1)

        return mean, variance, factor1

    def compute_predictive_likelihood(self, count, kernel_type, observations_kernel, optional_hyper_gp_obj):

        self.hyper_gp_obj = optional_hyper_gp_obj
        self.kernel_type = kernel_type
        K_x_x = self.compute_kernel_matrix_hyperkernel(self.X, self.X, observations_kernel)
        eye = 1e-1 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        L_x_x = np.linalg.cholesky(Knoise)
        K_x_xs = self.compute_kernel_matrix_hyperkernel(self.X, self.Xs, observations_kernel)
        factor1 = np.linalg.solve(L_x_x, K_x_xs)
        factor2 = np.linalg.solve(L_x_x, self.y)

        mean = np.dot(factor1.T, factor2).flatten()

        K_xs_xs = self.compute_kernel_matrix_hyperkernel(self.Xs, self.Xs, observations_kernel)
        variance_mat = K_xs_xs - np.dot(factor1.T, factor1)
        variance = np.diag(variance_mat)

        PH.printme(PH.p1,"\n\n\n\nMean: ", mean, "\nVariance: ", variance, "\nys: ",self.ys.shape,".....\n")

        neg_log_sum = 0
        for i in range(len(self.Xs)):

            likeli = (1/(np.sqrt(2*np.pi*variance[i])))*(np.exp((-1/(2*variance[i]))*((self.ys[i] - mean[i])**2)))
            if likeli == 0:
                PH.printme(PH.p4,
                           # "\n\n\n\nMean: ", mean, "\nVariance: ", variance,
                           "\nys shape: ", self.ys.shape, ".....\n")
                PH.printme(PH.p4,"Error occured in calculating likelihood ")
            neg_log_likelihood = -1 * np.log(likeli)
            neg_log_sum = neg_log_sum + neg_log_likelihood
        return np.array([neg_log_sum])


    def runGaussian(self, count, kernel_type, observations_kernel, optional_hyper_gp_obj):

        PH.printme(PH.p1, "!!!!!!!!!!Gaussian Process Started!!!!!!!!!")
        log_like_max = - 1 * float("inf")

        self.kernel_type = kernel_type
        if kernel_type == 'HYPER':
            self.hyper_gp_obj = optional_hyper_gp_obj
            likelihood = self.compute_log_marginal_likelihood_hyperkernel(observations_kernel)
            return likelihood

        if self.hyper_params_estimation:

            PH.printme(PH.p1, "Hyper Params estimating..")
            # Estimating Length scale itself
            x_max_value = None

            # Data structure to create the starting points for the scipy.minimize method
            random_points = []
            starting_points = []

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(self.number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(self.lengthscale_bounds[dim][0],
                                                               self.lengthscale_bounds[dim][1],
                                                               self.number_of_restarts_likelihood). \
                    reshape(1, self.number_of_restarts_likelihood)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points = np.vstack(random_points)

            # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
            for sample_num in np.arange(self.number_of_restarts_likelihood):
                array = []
                for dim_count in np.arange(self.number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                starting_points.append(array)
            starting_points = np.vstack(starting_points)

            variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                      self.signal_variance_bounds[1],
                                                      self.number_of_restarts_likelihood)

            total_bounds = self.lengthscale_bounds.copy()
            total_bounds.append(self.signal_variance_bounds)

            for ind in np.arange(self.number_of_restarts_likelihood):

                init_len_scale = starting_points[ind]
                init_var = variance_start_points[ind]

                init_points = np.append(init_len_scale, init_var)
                maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l(x),
                                      init_points,
                                      method='L-BFGS-B',
                                      tol=0.01,
                                      options={'maxfun': 20, 'maxiter': 20},
                                      bounds=total_bounds)

                len_scale_temp = maxima['x'][:self.number_of_dimensions]
                variance_temp = maxima['x'][len(maxima['x']) - 1]
                params = np.append(len_scale_temp, variance_temp)
                log_likelihood = self.optimize_log_marginal_likelihood_l(params)

                if (log_likelihood > log_like_max):
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for l= ",
                          maxima['x'][: self.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])
                    x_max_value = maxima
                    log_like_max = log_likelihood

            self.char_len_scale = x_max_value['x'][:self.number_of_dimensions]
            self.signal_variance = x_max_value['x'][len(maxima['x']) - 1]
            PH.printme(PH.p1, "Opt Length scale: ", self.char_len_scale, "\nOpt variance: ", self.signal_variance)

        if self.weight_params_estimation:

            x_max_value = None
            log_like_max = - 1 * float("inf")

            random_points_a = []
            random_points_b = []
            random_points_c = []
            random_points_d = []

            # Data structure to create the starting points for the scipy.minimize method
            random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[0][0],
                                                           self.len_weights_bounds[0][1],
                                                           self.number_of_restarts_likelihood).reshape(1,
                                                                                                       self.number_of_restarts_likelihood)
            random_points_a.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[1][0],
                                                           self.len_weights_bounds[1][1],
                                                           self.number_of_restarts_likelihood).reshape(1,
                                                                                                       self.number_of_restarts_likelihood)
            random_points_b.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[2][0],
                                                           self.len_weights_bounds[2][1],
                                                           self.number_of_restarts_likelihood).reshape(1,
                                                                                                       self.number_of_restarts_likelihood)
            random_points_c.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[3][0],
                                                           self.len_weights_bounds[3][1],
                                                           self.number_of_restarts_likelihood).reshape(1,
                                                                                                       self.number_of_restarts_likelihood)
            random_points_d.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_a = np.vstack(random_points_a)
            random_points_b = np.vstack(random_points_b)
            random_points_c = np.vstack(random_points_c)
            random_points_d = np.vstack(random_points_d)
            variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                      self.signal_variance_bounds[1],
                                                      self.number_of_restarts_likelihood)

            for ind in np.arange(self.number_of_restarts_likelihood):

                tot_init_points = []

                param_a = random_points_a[0][ind]
                tot_init_points.append(param_a)
                param_b = random_points_b[0][ind]
                tot_init_points.append(param_b)
                param_c = random_points_c[0][ind]
                tot_init_points.append(param_c)
                param_d = random_points_d[0][ind]
                tot_init_points.append(param_d)
                tot_init_points.append(variance_start_points[ind])
                total_bounds = self.len_weights_bounds.copy()
                total_bounds.append(self.signal_variance_bounds)

                maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_weight_params(x),
                                      tot_init_points,
                                      method='L-BFGS-B',
                                      tol=0.01,
                                      options={'maxfun': 20, 'maxiter': 20},
                                      bounds=total_bounds)

                params = maxima['x']
                log_likelihood = self.optimize_log_marginal_likelihood_weight_params(params)
                if log_likelihood > log_like_max:
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for params ", params)
                    x_max_value = maxima['x']
                    log_like_max = log_likelihood

            self.len_weights = x_max_value[0:4]
            self.signal_variance = x_max_value[len(maxima['x']) - 1]

            PH.printme(PH.p1, "Opt weights: ", self.len_weights, "   variance:", self.signal_variance)

        if self.degree_estimation:

            max_degree = 0
            log_like_max = - 1 * float("inf")

            for deg in np.arange(1, 11):
                log_likelihood = self.optimize_log_marginal_likelihood_degree(deg)
                PH.printme(PH.p1, "deg: ", deg, log_likelihood)
                if log_likelihood > log_like_max:
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for degree ", deg)
                    max_degree = deg
                    log_like_max = log_likelihood

            self.degree = max_degree

            PH.printme(PH.p1, "Opt Degree: ", self.degree)

        PH.printme(PH.p1, self.signal_variance)

        # compute the covariances between the test data points i.e K**
        K_xs_xs = self.computekernel(self.Xs, self.Xs)

        # Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
        L_xs_xs = np.linalg.cholesky(K_xs_xs + 1e-12 * np.eye(self.number_of_test_datapoints))

        # Sample 3 standard normals for our test points
        standard_normals = np.random.normal(size=(self.number_of_test_datapoints, 3))

        # multiply them by the square root of the covariance matrix
        f_prior = np.dot(L_xs_xs, standard_normals)
        mean, variance, factor1 = self.compute_mean_var(self.Xs, self.X, self.y)
        diag_variance = np.diag(variance)
        standard_deviation = np.sqrt(diag_variance)

        plot_posterior_distr_params = {'plotnum': 'Fig__' +self.output_gen_time+"__"+self.kernel_type,
                                       # 'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                                       # 'axis': [0, 1, self.linspaceymin, self.linspaceymax],
                                       'axis': [0, 1, 0, 1],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms20'], [self.Xs, self.ys, 'b-', 'label=True Fn'],
                                                      [self.Xs, mean, 'g--','label=Mean Fn','lw2']],
                                       'title': 'GP Posterior Distribution',
                                       'file': 'GP_Posterior_Distr'+ str(count),
                                       'gca_fill': [self.Xs.flat, mean - 2 * standard_deviation,
                                                    mean + 2 * standard_deviation] ,
                                       'xlabel': 'x',
                                       'ylabel': 'output, f(x)'
                                       }
        self.plot_graph(plot_posterior_distr_params)
        return log_like_max


if __name__ == "__main__":

    PH(os.getcwd())
    timenow = datetime.datetime.now()
    PH.printme(PH.p1, "\nStart time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    stamp = timenow.strftime("%H%M%S_%d%m%Y")

    kernel_type = 'SE'
    char_len_scale = 0.2
    number_of_test_datapoints = 500
    noise = 0.0
    random_seed = 500
    signal_variance = 1
    degree = 2

    # # Square wave
    # linspacexmin = 0
    # linspacexmax = 4
    # linspaceymin = -1.5
    # linspaceymax = 1.5

    # Oscillator
    # linspacexmin = 0
    # linspacexmax = 8
    # linspaceymin = 0
    # linspaceymax = 2.5

    # Complicated Oscillator
    # linspacexmin = 0
    # linspacexmax = 30
    # linspaceymin = 0
    # linspaceymax = 2

    #Benchmark
    # linspacexmin = 0
    # linspacexmax = 10
    # linspaceymin = -1.5
    # linspaceymax = 3

    # Levy
    # linspacexmin = -10
    # linspacexmax = 10
    # linspaceymin = -16
    # linspaceymax = 0

    # Triangular wave
    # linspacexmin = 0
    # linspacexmax = 10
    # linspaceymin = -1.5
    # linspaceymax = 1.5

    # Chirpwave wave
    # linspacexmin = 0
    # linspacexmax = 20
    # linspaceymin = -3
    # linspaceymax = 3

    #Sinc Mixture
    # linspacexmin = -15
    # linspacexmax = 15
    # linspaceymin = -0.5
    # linspaceymax = 1.5

    # # Gaussian Mixture
    # linspacexmin = 0
    # linspacexmax = 15
    # linspaceymin = -0.5
    # linspaceymax = 1.5

    # Linear
    # linspacexmin = 0
    # linspacexmax = 2
    # linspaceymin = 0
    # linspaceymax = 0.5

    # Linear Sin Function
    linspacexmin = 0
    linspacexmax = 10
    linspaceymin = 0
    linspaceymax = 10

    number_of_dimensions = 1
    number_of_observed_samples = 30
    hyper_params_estimation = False
    weight_params_estimation = False
    degree_estimation = False
    number_of_restarts_likelihood = 100
    oned_bounds = [[linspacexmin, linspacexmax]]
    sphere_bounds = [[linspacexmin, linspacexmax], [linspacexmin, linspacexmax]]
    michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
    random_bounds = [[0, 1], [1, 2]]
    # bounds = sphere_bounds
    bounds = oned_bounds
    # bounds = random_bounds

    Xmin = linspacexmin
    Xmax = linspacexmax
    ymax = linspaceymax
    ymin = linspaceymin

    a = 0.14
    b = 0.1
    lengthscale_bounds = [[0.1, 1]]
    signal_variance_bounds = [0.1, 1]
    true_func_type = "custom"
    fun_helper_obj = FunctionHelper(true_func_type)
    len_weights = [0.1, 0.3, 0.2]
    len_weights_bounds = [[0.1, 5] for i in range(4)]

    # Commenting for regression data - Forestfire
    random_points = []
    X = []

    # Generate specified (number of observed samples) random numbers for each dimension
    for dim in np.arange(number_of_dimensions):
        random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                       number_of_observed_samples).reshape(1, number_of_observed_samples)
        random_points.append(random_data_point_each_dim)

    # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
    random_points = np.vstack(random_points)

    # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
    for sample_num in np.arange(number_of_observed_samples):
        array = []
        for dim_count in np.arange(number_of_dimensions):
            array.append(random_points[dim_count, sample_num])
        X.append(array)
    X = np.vstack(X)

    # Sinc
    # x_obs = np.linspace(-15, -5,  20)
    # x_obs = np.append(x_obs, np.linspace(5, 15, 20))
    # X= x_obs.reshape(-1, 1)

    # Gaussian
    # x_obs = np.linspace(0, 5,  20)
    # x_obs = np.append(x_obs, np.linspace(10, 15, 20))
    # X= x_obs.reshape(-1, 1)

    # #Linear
    # # X = np.linspace(linspacexmin, linspacexmax, 10).reshape(-1, 1)
    # x_obs = np.linspace(linspacexmin, 0.5,  5)
    # x_obs = np.append(x_obs, np.linspace(1.5, linspacexmax, 5))
    # X= x_obs.reshape(-1, 1)

    # Linear Sin Function
    x_obs = np.linspace(linspacexmin, 3, 15)
    x_obs = np.append(x_obs, np.linspace(7, linspacexmax, 15))
    X = x_obs.reshape(-1, 1)


    y_arr = []
    for each_x in X:
        val = fun_helper_obj.get_true_func_value(each_x)
        y_arr.append(val)

    y = np.vstack(y_arr)
    # y =  sinc_function(X)

    X = np.divide((X - Xmin), (Xmax - Xmin))
    y = (y - ymin) / (ymax - ymin)

    random_points = []
    Xs = []

    # Generate specified (number of unseen data points) random numbers for each dimension
    for dim in np.arange(number_of_dimensions):
        random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],
                                                 number_of_test_datapoints).reshape(1,number_of_test_datapoints)
        random_points.append(random_data_point_each_dim)
    random_points = np.vstack(random_points)

    # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
    for sample_num in np.arange(number_of_test_datapoints):
        array = []
        for dim_count in np.arange(number_of_dimensions):
            array.append(random_points[dim_count, sample_num])
        Xs.append(array)
    Xs = np.vstack(Xs)
    # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently
    # Comented to accomodate Oscillator function mergen input domain
    # ys = fun_helper_obj.get_true_func_value(Xs)

    ys_arr = []
    for each_xs in Xs:
        val_xs = fun_helper_obj.get_true_func_value(each_xs)
        ys_arr.append(val_xs)

    ys = np.vstack(ys_arr)
    # ys = sinc_function(Xs)

    Xs = np.divide((Xs - Xmin), (Xmax - Xmin))
    ys = (ys - ymin) / (ymax - ymin)

    gaussianObject = GaussianProcess(str(stamp), kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                     linspacexmax, linspaceymin, linspaceymax, signal_variance,
                                     number_of_dimensions, number_of_observed_samples, X, y, hyper_params_estimation,
                                     number_of_restarts_likelihood, lengthscale_bounds, signal_variance_bounds, fun_helper_obj, Xmin,
                                     Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights, len_weights_bounds, weight_params_estimation,
                                     degree_estimation, degree)


    count = 1

    PH.printme(PH.p1, "kernel_type: ", kernel_type, "\tnumber_of_test_datapoints: ",  number_of_test_datapoints, "\tnoise:", noise,  "\trandom_seed:",
          random_seed,  "\nlinspacexmin:", linspacexmin,"\tlinspacexmax:", linspacexmax, "\tlinspaceymin:",  linspaceymin,
          "\tlinspaceymax:",    linspaceymax, "\nsignal_variance:",  signal_variance, "\tnumber_of_dimensions:", number_of_dimensions,
          "\tnumber_of_observed_samples:", number_of_observed_samples,  "\nhyper_params_estimation:",  hyper_params_estimation,
          "\tnumber_of_restarts_likelihood:", number_of_restarts_likelihood, "\tlengthscale_bounds:",
          lengthscale_bounds, "\tsignal_variance_bounds:",    signal_variance_bounds,   "\nXmin:", Xmin, "\tXmax:", Xmax, "\tymin:", ymin,
          "\tymax:", ymax, "\tchar_len_scale:", char_len_scale, "\tlen_weights:", len_weights, "\tlen_weights_bounds:",
          len_weights_bounds, "\tweight_params_estimation:", weight_params_estimation, "\nX:", X, "\ty:", y)

    # kernel_types = ['SE', 'MATERN3', 'MKL']
    # kernel_types = ['LIN', 'PER', 'POLY', 'MKL']
    kernel_types = ['PER']
    for kernel in kernel_types:
        PH.printme(PH.p1, "\n\nKernel: ", kernel)
        if kernel == 'MKL':
            gaussianObject.weight_params_estimation = True
            gaussianObject.hyper_params_estimation = False
            gaussianObject.degree_estimation = False
        elif kernel == "POLY":
            gaussianObject.weight_params_estimation = False
            gaussianObject.hyper_params_estimation = False
            gaussianObject.degree_estimation = True
        else:
            gaussianObject.hyper_params_estimation = True
        gaussianObject.runGaussian(count, kernel, None, None)

    timenow = datetime.datetime.now()
    PH.printme(PH.p1, "\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    plt.show()

