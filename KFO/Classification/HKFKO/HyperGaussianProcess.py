import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import scipy.optimize as opt
from scipy.stats import norm

import scipy.optimize as opt
import scipy as sp
import matplotlib.pyplot as plt
# from numpy import *
import numpy as np
import datetime
from scipy.spatial import distance
import collections
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
# from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH



class HyperGaussianProcess:

    def __init__(self, X, hyperkernel_type, kernel_type, char_length_scale, sigma_len, sigma_len_bounds, signal_variance_bounds,
                 number_of_samples_in_X_for_grid,
                 number_of_samples_in_Xs_for_grid, number_of_test_datapoints, noise, hyper_lambda,
                 random_seed, max_X, min_X, max_Y, min_Y, bounds,
                 number_of_dimensions, signal_variance, number_of_basis_vectors_chosen, basis_weights_bounds, len_scale_bounds,
                 number_of_restarts_likelihood, no_principal_components, hyper_char_len_scale):

        self.X = X
        self.hyperkernel_type = hyperkernel_type
        self.kernel_type = kernel_type
        self.char_length_scale = char_length_scale
        self.sigma_len = sigma_len
        self.sigma_len_bounds = sigma_len_bounds
        self.signal_variance_bounds = signal_variance_bounds
        self.number_of_samples_in_X_for_grid = number_of_samples_in_X_for_grid
        self.number_of_samples_in_Xs_for_grid = number_of_samples_in_Xs_for_grid
        self.number_of_test_datapoints = number_of_test_datapoints
        self.noise = noise
        self.hyper_lambda = hyper_lambda
        self.random_seed = random_seed
        self.max_X = max_X
        self.min_X = min_X
        self.max_Y = max_Y
        self.min_Y = min_Y
        self.bounds = bounds
        self.number_of_dimensions = number_of_dimensions
        self.kappa_kernel_Xtil_Xtil = None
        self.signal_variance = signal_variance
        self.number_of_basis_vectors_chosen = number_of_basis_vectors_chosen
        self.observations_kernel = []
        self.observations_y = []
        self.basis_weights_bounds = basis_weights_bounds
        self.current_kernel_samples = None
        self.current_kernel_bias = None
        self.best_kernel = None
        self.L_kappa_kernel_Xtil_Xtil = None
        self.L_K_K_hypergp = None
        self.pre_calculated_L_Kappa_Kobs = None  # Not required with EVD method of computations
        self.principal_eigen_values = None
        self.principal_eigen_vectors = None
        self.eigen_diag = None
        self.len_scale_bounds = len_scale_bounds
        self.number_of_restarts_likelihood = number_of_restarts_likelihood
        self.no_principal_components = no_principal_components
        self.hyper_char_len_scale = hyper_char_len_scale

    def gaussian_harmonic_hyperkernel(self, datapoint1, datapoint2, datapoint3, datapoint4):

        num = 1 - self.hyper_lambda
        difference1 = datapoint1 - datapoint2
        difference2 = datapoint3 - datapoint4
        den = 1 - (self.hyper_lambda * (np.exp(-1 * (self.sigma_len ** 2) * ((np.dot(difference1, difference1.T)) + (np.dot(
            difference2, difference2.T))))))
        kernel_val = num / den
        return kernel_val

    def matern_harmonic_hyperkernel(self, datapoint1, datapoint2, datapoint3, datapoint4):

        num = 1 - self.hyper_lambda
        difference1 = datapoint1 - datapoint2
        l2_difference1 = np.sqrt(np.dot(difference1, difference1.T))
        difference2 = datapoint3 - datapoint4
        l2_difference2 = np.sqrt(np.dot(difference2, difference2.T))

        factor1 = (1+((np.sqrt(3)/self.hyper_char_len_scale)*l2_difference1))*(1+((np.sqrt(3)/self.hyper_char_len_scale)*l2_difference2))
        factor2 = np.exp((-1 * np.sqrt(3) / self.hyper_char_len_scale) * (l2_difference1 + l2_difference2))

        den = 1 - (self.hyper_lambda * factor1 * factor2)
        kernel_val = num / den
        return kernel_val

    def polynomial_kernel(self, datapoint1, datapoint2, datapoint3, datapoint4):
        num = 1 - self.hyper_lambda
        prod1 = np.dot(datapoint1, datapoint2.T)
        prod2 = np.dot(datapoint3, datapoint4.T)
        val = ((1 + prod1 + prod2 + np.dot(prod1, prod2))*(1 + prod1 + prod2 + np.dot(prod1, prod2)))
        val = val / (((1 + self.number_of_dimensions)*(1+ self.number_of_dimensions))*((1 + self.number_of_dimensions)*(1+ self.number_of_dimensions)))
        den = 1 - (self.hyper_lambda * val)
        kernel_val = num / den
        return kernel_val

    def linear_kernel(self, datapoint1, datapoint2, datapoint3, datapoint4):
        num = 1 - self.hyper_lambda
        prod1 = np.dot(datapoint1, datapoint2.T)
        prod2 = np.dot(datapoint3, datapoint4.T)
        val = ((1 + prod1 + prod2 + np.dot(prod1, prod2)))
        den = 1 - (self.hyper_lambda * val)
        kernel_val = num / den
        return kernel_val

    def free_four_kernel(self, datapoint1, datapoint2, datapoint3, datapoint4):
        num = 1 - self.hyper_lambda
        prod1 = np.dot(datapoint1, datapoint2)
        prod2 = np.dot(prod1, datapoint3)
        prod3 = np.dot(prod2, datapoint4)
        prod3 = prod3/(self.number_of_dimensions*4)
        den = 1 - (self.hyper_lambda * prod3)
        kernel_val = num / den
        return kernel_val

    def free_rbf_kernel(self, datapoint1, datapoint2, datapoint3, datapoint4):

        num = 1 - self.hyper_lambda
        prod1 = np.multiply(datapoint1, datapoint2)
        prod2 = np.multiply(datapoint3, datapoint4)
        tot_prod = np.multiply(prod1, prod2)
        term1 = np.sum(tot_prod)
        term2 = (2 * term1 - (datapoint1 ** 2) - (datapoint2 ** 2) - (datapoint3 ** 2) - (datapoint4**2))
        val = np.exp(0.5 * (1 / (self.hyper_char_len_scale*self.hyper_char_len_scale)) * term2)
        den = 1 - (self.hyper_lambda * val)
        kernel_val = num / den
        return kernel_val


    def compute_kernel_with_hyperkernel(self, x1, x2, x3, x4):
        if self.hyperkernel_type == 'gaussian_harmonic_kernel':
            covariance = self.gaussian_harmonic_hyperkernel(x1, x2, x3, x4)

        elif self.hyperkernel_type == 'matern_harmonic_kernel':
            covariance = self.matern_harmonic_hyperkernel(x1, x2, x3, x4)

        elif self.hyperkernel_type == 'polynomial_kernel':
            covariance = self.polynomial_kernel(x1, x2, x3, x4)

        elif self.hyperkernel_type == 'free_four_kernel':
            covariance = self.free_four_kernel(x1, x2, x3, x4)

        elif self.hyperkernel_type == 'free_rbf_kernel':
            covariance = self.free_rbf_kernel(x1, x2, x3, x4)

        elif self.hyperkernel_type == 'linear_kernel':
            covariance = self.linear_kernel(x1, x2, x3, x4)

        return covariance

    def compute_covariance_in_X2(self, X):

        # #old method
        total_possible_sample_X_tils = len(X) * len(X)
        kappa_kernel_Xtil_Xtil = np.zeros(shape=(total_possible_sample_X_tils, total_possible_sample_X_tils))
        for kap_i in range(0, total_possible_sample_X_tils):
            for kap_j in range(0, total_possible_sample_X_tils):
                index1 = int(kap_i / len(X))
                index2 = int(kap_j / len(X))
                index3 = kap_i % len(X)
                index4 = kap_j % len(X)
                kappa_kernel_Xtil_Xtil[kap_i][kap_j] = self.compute_kernel_with_hyperkernel(X[index1], X[index2], X[index3], X[index4])
        return kappa_kernel_Xtil_Xtil


    def obtain_mercer_from_krien(self, standard_normals):

        # #To convert Krien kernel to Mercer kernel
        mercer_kernel_samples = []
        krien_kernel_vector = np.dot(self.sqrt_kappa, standard_normals)
        for each_krien_kernel_index in range(self.number_of_basis_vectors_chosen):
            krien_kernel_matrix = krien_kernel_vector[:, each_krien_kernel_index].reshape(self.number_of_samples_in_X_for_grid,
                                                                                          self.number_of_samples_in_X_for_grid)
            krien_eigen_values, krien_eigen_vectors = np.linalg.eigh(krien_kernel_matrix)

            # Spectrum Flip
            for kri_index in range(len(krien_eigen_values)):
                if krien_eigen_values[kri_index] < 0:
                    krien_eigen_values[kri_index] = abs(krien_eigen_values[kri_index])

            krien_upd_eig_diag = np.diag(krien_eigen_values)
            product1 = np.dot(krien_eigen_vectors, krien_upd_eig_diag)
            updated_krien_matrix = np.dot(product1, krien_eigen_vectors.T)
            updated_krien_vector = updated_krien_matrix.reshape(self.number_of_samples_in_X_for_grid * self.number_of_samples_in_X_for_grid,
                                                                1)

            product1 = np.dot(self.inv_sqrt_eigen_diag, self.principal_eigen_vectors.T)
            updated_mercer_kernel_sample = np.dot(product1, updated_krien_vector)
            mercer_kernel_samples.append(updated_mercer_kernel_sample)

        # Fix reshape, check negative values, check inv_sqrt_kappa
        mercer_kernel_samples = np.hstack(mercer_kernel_samples)
        mercer_kernel_samples = mercer_kernel_samples.reshape(self.no_principal_components, self.number_of_basis_vectors_chosen)
        return mercer_kernel_samples.reshape(self.no_principal_components, self.number_of_basis_vectors_chosen)


    def compute_kappa_utils(self):

        PH.printme(PH.p2, "Eigen values for the calculation of the dimensionality reduction...")

        # # # New method with Eigen values and Eigen vectors
        self.kappa_kernel_Xtil_Xtil = self.compute_covariance_in_X2(self.X)
        eigen_values, eigen_vectors = np.linalg.eigh(self.kappa_kernel_Xtil_Xtil)

        # Compute the principal components
        total_components = len(eigen_values)

        #New method to fetch descending order values and vectors
        prin_eig_vecs = []
        prin_eig_vals = []

        for index_PC in range(total_components, total_components-self.no_principal_components, -1):
            prin_eig_vals.append(eigen_values[index_PC-1])
            prin_eig_vecs.append((eigen_vectors[:, index_PC-1]))

        self.principal_eigen_values = np.hstack(prin_eig_vals)
        self.principal_eigen_vectors = np.array(prin_eig_vecs).T
        self.eigen_diag = np.diag(self.principal_eigen_values)

        # Further computations like Square root of the kappa matrix
        self.eigen_diag = np.diag(self.principal_eigen_values)
        self.sqrt_eigen_diag = np.sqrt(self.eigen_diag)
        self.inv_sqrt_eigen_diag = np.linalg.inv(self.sqrt_eigen_diag)
        self.inv_eigen_diag = np.linalg.inv(self.eigen_diag)
        self.sqrt_kappa = np.dot(self.principal_eigen_vectors, self.sqrt_eigen_diag)
        self.inv_kappa_matrix = np.dot(self.principal_eigen_vectors, np.dot(self.inv_eigen_diag, self.principal_eigen_vectors.T))


    def generate_basis_as_kernel_samples(self):

        standard_normals = np.random.normal(size=(self.no_principal_components, self.number_of_basis_vectors_chosen))
        kernel_samples = standard_normals.T
        return kernel_samples

    def SE_Kernel_gnorm(self, data_point1, data_point2, char_length_scale, signal_variance):
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                L_sol_dp1 = np.linalg.solve(self.L_kappa_kernel_Xtil_Xtil, data_point1[i, :].T)
                alpha1 = np.linalg.solve(self.L_kappa_kernel_Xtil_Xtil.T, L_sol_dp1)
                L_sol_dp2 = np.linalg.solve(self.L_kappa_kernel_Xtil_Xtil, data_point2[j, :].T)
                alpha2 = np.linalg.solve(self.L_kappa_kernel_Xtil_Xtil.T, L_sol_dp2)
                alpha_difference = alpha1 - alpha2
                sq_difference = np.dot(alpha_difference.T, np.dot(self.kappa_kernel_Xtil_Xtil, alpha_difference))
                each_kernel_val = (signal_variance ** 2) * np.exp(-0.5 * (1 / (char_length_scale ** 2)) * sq_difference)
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def SE_Kernel_l2(self, data_point1, data_point2, char_length_scale, signal_variance):
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                alpha_difference = data_point1[i, :] - data_point2[j, :]
                sq_difference = np.dot(alpha_difference.T, alpha_difference)
                each_kernel_val = (signal_variance ** 2) * np.exp(-0.5 * (1 / (char_length_scale ** 2)) * sq_difference)
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def compute_hyperparams_kernel_observations(self):
        PH.printme(PH.p1, "Hyper Params for kernel observations")

        log_like_max = - 1 * float("inf")
        # Estimating Length scale itself
        x_max_value = None

        # Data structure to create the starting points for the scipy.minimize method
        random_points = []
        starting_points = []

        # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
        for dim in np.arange(self.number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(self.len_scale_bounds[dim][0],
                                                           self.len_scale_bounds[dim][1],
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

        total_bounds = self.len_scale_bounds.copy()
        total_bounds.append(self.signal_variance_bounds)

        for ind in np.arange(self.number_of_restarts_likelihood):

            init_len_scale = starting_points[ind]
            init_var = variance_start_points[ind]

            init_points = np.append(init_len_scale, init_var)
            # init_points = init_len_scale      #used if only lengthscale has to be tuned
            maxima = opt.minimize(lambda x: -self.opt_log_likeli_kernel_l(x),
                                  init_points,
                                  method='L-BFGS-B',
                                  tol=0.01,
                                  options={'maxfun': 20, 'maxiter': 20},
                                  bounds=total_bounds)

            len_scale_temp = maxima['x'][:self.number_of_dimensions]
            variance_temp = maxima['x'][len(maxima['x']) - 1]
            params = np.append(len_scale_temp, variance_temp)
            # params = len_scale_temp #used if only lengthscale has to be tuned
            log_likelihood = self.opt_log_likeli_kernel_l(params)

            if log_likelihood > log_like_max:
                PH.printme(PH.p2, "New maximum log likelihood ", log_likelihood, " found for l= ",
                      maxima['x'][: self.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])
                x_max_value = maxima
                log_like_max = log_likelihood
        self.char_length_scale = x_max_value['x'][:self.number_of_dimensions]
        self.signal_variance = x_max_value['x'][len(maxima['x']) - 1]
        PH.printme(PH.p2, "Opt Length scale: ", self.char_length_scale, "\nOpt variance: ", self.signal_variance)

    def opt_log_likeli_kernel_l(self, input):

        init_charac_length_scale = np.array(input[: self.number_of_dimensions])
        signal_variance = input[len(input) - 1]
        K_x_x = self.SE_Kernel_l2(self.observations_kernel, self.observations_kernel, init_charac_length_scale, signal_variance)
        eye = 1e-6 * np.eye(len(self.observations_kernel))
        Knoise = K_x_x + eye
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.observations_y)
        log_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          len(self.observations_kernel) * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_likelihood


    def compute_covariance_matrix_for_kernels(self, data_point1, data_point2):

        # Kernel_type = SE represents the Squared Exponential Kernel
        if self.kernel_type == "SE":
            result = self.SE_Kernel_l2(data_point1, data_point2, self.char_length_scale, self.signal_variance)

            # Kernel_type = 1 represents the Rational Quadratic Function Kernel
        elif self.kernel_type == 1:
            PH.printme(PH.p1, "RQF Kernel")
            alpha = 0.1
            result = self.rational_quadratic_kernel(data_point1, data_point2, self.char_length_scale, alpha)

        return result

    def predict_for_hypergp(self, lambda_weights):

        # # Eigen way to predict
        k_new = np.zeros(shape=(len(lambda_weights), self.no_principal_components))
        for i in range(len(lambda_weights)):
            k_new[i] = self.current_kernel_bias + lambda_weights[i][0] * self.current_kernel_samples[0]
        cov_K_xs_x = self.compute_covariance_matrix_for_kernels(self.observations_kernel, k_new)
        factor1 = np.linalg.solve(self.L_K_K_hypergp, cov_K_xs_x)
        factor2 = np.linalg.solve(self.L_K_K_hypergp, self.observations_y)
        mean = np.dot(factor1.T, factor2)

        cov_K_xs_xs = self.compute_covariance_matrix_for_kernels(k_new, k_new)
        variance = cov_K_xs_xs - np.dot(factor1.T, factor1)
        return mean, variance

    def estimate_kernel_for_Xtil(self, datapointx1, datapointx2, current_observations_kernel):

        kernel_mat = []
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                hyper_value_num = self.compute_kernel_with_hyperkernel(datapointx1, datapointx2, self.X[i], self.X[j])
                kernel_mat = np.append(kernel_mat, hyper_value_num)
        current_observations_with_eigen = np.dot(self.sqrt_kappa, current_observations_kernel[0].T)
        estimated_kernel_value = np.dot(kernel_mat,  np.dot(self.inv_kappa_matrix, current_observations_with_eigen))
        return estimated_kernel_value

    def plot_kernel_wrapper(self, current_observations_kernel, msg):

        PH.printme(PH.p1, "plotting kernel\n",current_observations_kernel)
        self.kappa_kernel_Xtil_Xtil = self.compute_covariance_in_X2(self.X)

        # Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
        self.L_kappa_kernel_Xtil_Xtil = np.linalg.cholesky(self.kappa_kernel_Xtil_Xtil + 1e-6 * np.eye(len(self.X) * len(self.X)))

        kernel_mat = np.zeros(shape=(100, 100))
        xbound = np.linspace(0, 1, 100).reshape(-1, 1)
        X1, X2 = np.meshgrid(xbound, xbound)
        for xb_i in range(len(xbound)):
            for xb_j in range(len(xbound)):
                kernel_mat[xb_i][xb_j] = self.estimate_kernel_for_Xtil_plot(xbound[xb_i], xbound[xb_j], current_observations_kernel)

        fig = plt.figure(msg)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X1, X2, kernel_mat, rstride=1, cstride=1,
                               cmap='viridis', linewidth=1, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=20)



    def estimate_kernel_for_Xtil_plot(self, datapointx1, datapointx2, current_observations_kernel):

        kernel_mat = []
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                hyper_value = self.compute_kernel_with_hyperkernel(datapointx1, datapointx2, self.X[i], self.X[j])
                kernel_mat = np.append(kernel_mat, hyper_value)

        L_sol_ker_obs_old = np.linalg.solve(self.L_kappa_kernel_Xtil_Xtil, current_observations_kernel[0].T)
        L_sol_ker_new_obs_old = np.linalg.solve(self.L_kappa_kernel_Xtil_Xtil, kernel_mat)
        estimated_kernel_value = np.dot(L_sol_ker_new_obs_old.T, L_sol_ker_obs_old)

        return estimated_kernel_value
