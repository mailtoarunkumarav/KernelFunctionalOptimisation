from scipy.stats import norm
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH

# Class to handle the Acquisition Functions related tasks required for the Bayesian Optimization
class AcquisitionUtility():

    # Initializing the parameters required for the ACQ functions
    def __init__(self, acq_type, number_of_restarts, extrema_type):
        self.acq_type = acq_type
        self.number_of_restarts = number_of_restarts
        self.extrema_type = extrema_type
        self.hyper_gaussian_object = None

    # Method to set the type of ACQ function to be used for the Optimization process
    def set_acq_func_type(self, acq_type):
        self.acq_type = acq_type

    # UCB ACQ function
    def ucb_acq_func(self, mean, std_dev, iteration_count):

        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):

            # Constant parameters to be used while maximizing the ACQ function
            delta = 0.1
            v = 1
            d = self.hyper_gaussian_object.number_of_dimensions
            beta3 = 2 * np.log((iteration_count**((d/2)+2)) * (np.pi**2) * (1/(3*delta)))
            gp_ucb_kappa = np.sqrt(v * beta3)
            result = mean + gp_ucb_kappa * std_dev
            return result

    def upper_confidence_bound_util(self, lambda_weights, hyper_gp_obj, iteration_count):

        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance = hyper_gp_obj.predict_for_hypergp(np.array([lambda_weights]))
            std_dev = np.sqrt(variance)
            result = self.ucb_acq_func(mean, std_dev, iteration_count)
            return -1 * result

    def maximise_acq_function(self, hyper_gaussian_object, iteration_count):

        self.hyper_gaussian_object = hyper_gaussian_object
        # Initialize the xmax value and the function values to zeroes

        max_basis_weights = np.zeros(self.hyper_gaussian_object.number_of_basis_vectors_chosen)
        basis_weights_bounds = [self.hyper_gaussian_object.basis_weights_bounds for count in
                                range(self.hyper_gaussian_object.number_of_basis_vectors_chosen)]
        function_best = self.extrema_type * float("inf")

        # Temporary data structures to store xmax's, function values of each run of finding maxima using scipy.minimize
        tempweights_lambda = []
        temp_fvals = []

        # Data structure to create the starting points for the scipy.minimize method
        random_starting_points = []
        # Depending on the number of dimensions and bounds, generate random multiple starting points to find maxima
        for restart_count in np.arange(self.number_of_restarts):
            random_data_point_each_dim = np.random.uniform(self.hyper_gaussian_object.basis_weights_bounds[0],
                                                           self.hyper_gaussian_object.basis_weights_bounds[1],
                                                           self.hyper_gaussian_object.number_of_basis_vectors_chosen)
            random_starting_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_starting_points = np.vstack(random_starting_points)

        if self.acq_type == "UCB":

            PH.printme(PH.p2, "ACQ Function : UCB ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in random_starting_points:
                # Find the maxima in the bounds specified for the UCB ACQ function
                max_weights = opt.minimize(lambda x: self.upper_confidence_bound_util(x, self.hyper_gaussian_object,  iteration_count),
                                           starting_point, method='L-BFGS-B', tol=0.001, bounds=basis_weights_bounds)

                mean, variance = self.hyper_gaussian_object.predict_for_hypergp(np.array([max_weights['x']]))
                std_dev = np.sqrt(variance)
                fvalue = self.ucb_acq_func(mean, std_dev, iteration_count)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempweights_lambda.append(max_weights['x'])
                temp_fvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if fvalue > function_best:
                    PH.printme(PH.p2, "New best Fval: ", fvalue, " found at: ", max_weights['x'])
                    max_basis_weights = max_weights['x']
                    function_best = fvalue

            PH.printme(PH.p2, "UCB Best is ", function_best, "at ", max_basis_weights)

            # # ##Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            # with np.errstate(invalid='ignore'):
            #     mean, variance = self.hyper_gaussian_object.predict_for_hypergp()
            #     std_dev = np.sqrt(variance)
            #     acq_func_values = self.ucb_acq_func(mean, std_dev, iteration_count)

        return max_basis_weights

    # Helper method to plot the values found for the specified ACQ function at unseen data points
    def plot_acquisition_function(self, count, Xs, acq_func_values, plot_axis):

        # Set the parameters of the ACQ functions plot
        plt.figure('Acquisition Function - ' + str(count))
        plt.clf()
        plt.plot(Xs, acq_func_values)
        plt.axis(plot_axis)
        plt.title('Acquisition Function')
        plt.savefig('acq', bbox_inches='tight')

    def plot_graph(self, count, Xs, len_values, plot_axis):

        # Set the parameters of the ACQ functions plot
        plt.figure('lengthscale - ' + str(count))
        plt.clf()
        plt.plot(Xs, len_values)
        # plt.axis(plot_axis)
        plt.title('lengthscale')
        plt.savefig('len'+str(count), bbox_inches='tight')


