from HyperGaussianProcess import HyperGaussianProcess
from AcquisitionUtility import AcquisitionUtility
from KernelOptimiser import KernelOptimiser
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys, getopt
from SVM_Wrapper import SVM_Wrapper

import os
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH



# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams["font.size"] = 12
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')

# To fix the random number genration, currently not able, so as to retain the random selection of points
random_seed = 400
np.random.seed(random_seed)


# Class for starting Bayesian Optimization with the specified parameters
class KernelOptimizationWrapper:

    def kernel_wrapper(self, start_time, input, cmd_inputs):

        # Number of data points in X  for the grid
        number_of_samples_in_X_for_grid = 4
        number_of_samples_in_Xs_for_grid = 5

        # Lower bound considered, required for sample generation
        min_X = 0
        # Upper bound considered, required for sample generation
        max_X = 1
        # Minimum value for the objective function
        min_Y = 1
        # Maximum value for the objective function
        max_Y = 2

        number_of_dimensions = 1
        bounds = [[min_X, max_X] for d in range(number_of_dimensions)]

        # Number of points required to be observed to evaluate the unknown function ((10:20)*no_dimensions )
        # number_of_iterations = number_of_dimensions * 10

        if "subspaces" in cmd_inputs:
            number_of_subspace_selection_iterations = cmd_inputs["subspaces"]
        else:
            number_of_subspace_selection_iterations = 5

        if "iterations" in cmd_inputs:
            number_of_iterations_best_solution = cmd_inputs["iterations"]
        else:
            number_of_iterations_best_solution = 5



        number_of_basis_vectors_chosen = 1

        basis_weights_bounds = [0.1, 1]

        number_of_init_random_kernel_y_observations = 4

        # extrema_type = -1 if function maxima is considered
        # extrema_type = 1  if function minima is considered Ex: Regret is +infinity, as Regret reduces over iterations
        extrema_type = -1

        # Number of samples for which we have the y values to be known (no_dimensions+1)
        # number_of_observed_samples = number_of_dimensions + 1

        # Number of restarts required during the calculation of the maxima during the maximization of acq functions
        number_of_restarts_acq = 100

        # Number of restarts required during the calculation of the maxima during the maximization of likelihood
        number_of_restarts_likelihood = 100

        # Number of runs the BO has to be run for calculating the regret and the corresponding means and the standard devs

        if "runs" in cmd_inputs:
            number_of_runs = cmd_inputs["runs"]
        else:
            number_of_runs = 5

        # Type of kernel to be used in the optimization process
        # 0 - Squared Exponential; 1 - Rational Quadratic Function; 2 - Exponential; 3 - Periodic***(To be fixed)
        # hyperkernel_type = 'gaussian_harmonic_kernel'
        hyperkernel_type = 'matern_harmonic_kernel'
        # hyperkernel_type = 'polynomial_kernel'
        # hyperkernel_type = 'free_four_kernel'
        # hyperkernel_type = 'free_rbf_kernel'
        # hyperkernel_type = 'linear_kernel'


        if input is not None:
            hyper_lambda = input[0]
            hyper_char_len_scale = input[1]
            PH.printme(PH.p1, "Inputs Supplied: ", input)

        else:
            hyper_lambda = 0.01
            # Matern Harmonic Kernel
            hyper_char_len_scale = 0.0105

        # Characteristics of the Kernel being used, currently implemented for the square exponential kernel
        # Required to set up the GP Object initially
        kernel_char = 'ard'

        # characteristic_length_scale = np.array([1 for nd in np.arange(number_of_dimensions)])
        # char_length_scale = [1 for nd in np.arange(number_of_dimensions)]
        char_length_scale = 0.3
        len_scale_bounds = [[0.1, 1]]

        # sigma_len = 0.0001
        sigma_len = 0.0001
        sigma_len_bounds = [0.1, 1]

        # Signal variance bounds
        signal_variance_bounds = [0.1, 1]

        # Initial Signal Variance
        signal_variance = 1

        # Number of unseen data points in order to evaluate the function
        number_of_test_datapoints = 100

        # Noise to be added in the system***
        noise = 0.0

        # Number of principal components
        no_principal_components = 10

        # Data structure to hold the regrets obtained from each type of ACQ function in each run of the BO
        total_ucb_regret = []
        total_ei_regret = []
        total_pi_regret = []
        total_rs_regret = []

        # Added to generate the results for comparing regrets for spatially varying length scales.
        ei_ard_regret = []
        ei_var_l_regret = []
        ei_fixed_l_regret = []
        ei_multi_l_regret = []

        # kernel_iter_types = ['var_l', 'm_ker', 'ard', 'fix_l']
        kernel_iter_types = ['fix_l']
        # kernel_iter_types = ['m_ker']

        tot_max_acc = np.array([])
        tot_best_sol = np.array([])

        kernel_type = "SE"

        # Initial value to denote the type of ACQ function to be used, but ignored as all ACQs are run in the sequence
        # acq_fun_list = ['ei', 'pi', 'rs', 'ucb']
        acq_fun_list = ['UCB']


        if "dataset" in cmd_inputs:
            dataset = cmd_inputs["dataset"]
        else:
            # dataset = "wdbc"
            # dataset = "iris"
            # dataset = "glass"
            # dataset = "ionos"
            # dataset = "sonar"
            # dataset = "heart"
            # dataset = "credit"
            # dataset = "credit_arc"
            # dataset = "seeds"
            # dataset = "pima"
            # dataset = "dermatology"
            dataset = "wine"
            # dataset = "bio"
            # dataset = "contra"
            # dataset = "pho"
            # dataset = "hay"
            # dataset = "eco"
            # dataset = "car"

        PH.printme(PH.p1, "Configuration Settings:\n\n", "hyperkernel_type:", hyperkernel_type, "\tkernel_type:", kernel_type, "\tchar_length_scale:",
              char_length_scale, "\tsigma_len:", sigma_len, "\tsigma_len_bounds: ", sigma_len_bounds,
              # "gnorm",
              "\tsignal_variance_bounds:", signal_variance_bounds, "\nnumber_of_samples_in_X_for_grid:",
              number_of_samples_in_X_for_grid, "\tnumber_of_samples_in_Xs_for_grid:", number_of_samples_in_Xs_for_grid,
              "\tnumber_of_test_datapoints:", number_of_test_datapoints, "\tnoise:", noise, "\nhyper_lambda", hyper_lambda,
              "\trandom_seed:", random_seed, "\tmax_X:", max_X, "\tmin_X:", min_X, "\tmax_Y:", max_Y, "\tmin_Y:", min_Y, "\nbounds:",
              bounds, "\tnumber_of_dimensions:", number_of_dimensions, "\nsignal_variance:", signal_variance,
              "\tnumber_of_basis_vectors_chosen: ", number_of_basis_vectors_chosen, "\tbasis_weights_bounds:", basis_weights_bounds,
              "\nnumber_of_subspace_selection_iterations:", number_of_subspace_selection_iterations,
              "\tnumber_of_iterations_best_solution:", number_of_iterations_best_solution,
              "\nnumber_of_init_random_kernel_y_observations:", number_of_init_random_kernel_y_observations, "\tacq_fun_list:",
              acq_fun_list, "\nLength scale bounds: ", len_scale_bounds, "\tnumber of restart likelihoods: ",
              number_of_restarts_likelihood, "\thyper_char_len_scale: ", hyper_char_len_scale, "\n\nDataset:", dataset,"\n clip_indicator")


        PH.printme(PH.p1, "\n###################################################################\n")
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for i in range(number_of_runs):

            X = []

            # updated implementation
            random_points = []
            # Generate specified (number of observed samples) random numbers for each dimension
            for dim in np.arange(number_of_dimensions):
                # random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1], number_of_observed_samples).reshape(1,
                #                                                                                     number_of_observed_samples)

                random_data_point_each_dim = np.linspace(min_X, max_X, number_of_samples_in_X_for_grid).reshape(1,
                                                                                                        number_of_samples_in_X_for_grid)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
            random_points = np.vstack(random_points)

            # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
            for sample_num in np.arange(number_of_samples_in_X_for_grid):
                array = []
                for dim_count in np.arange(number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                X.append(array)
            X = np.vstack(X)

            # Create Gaussian Process  object to carry on the prediction and model fitting tasks
            hyper_gaussian_object = HyperGaussianProcess(X, hyperkernel_type, kernel_type, char_length_scale, sigma_len, sigma_len_bounds,
                                                         signal_variance_bounds, number_of_samples_in_X_for_grid,
                                                         number_of_samples_in_Xs_for_grid,
                                                         number_of_test_datapoints, noise, hyper_lambda,
                                                         random_seed, max_X, min_X, max_Y, min_Y, bounds,
                                                         number_of_dimensions, signal_variance, number_of_basis_vectors_chosen,
                                                         basis_weights_bounds, len_scale_bounds, number_of_restarts_likelihood,
                                                         no_principal_components, hyper_char_len_scale)

            acquisition_utility_object = AcquisitionUtility(None, number_of_restarts_acq, extrema_type)

            svm_wrapper_obj = SVM_Wrapper()
            svm_wrapper_obj.construct_svm_classifier(dataset)

            kernel_optimiser_obj = KernelOptimiser(hyper_gaussian_object, acquisition_utility_object,
                                                   number_of_subspace_selection_iterations, number_of_iterations_best_solution,
                                                   number_of_init_random_kernel_y_observations, svm_wrapper_obj)

            # #Algorithm running for PI acquisition function
            if 'UCB' in acq_fun_list:
                acq_type = 'UCB'
                kernel_optimiser_obj.acquisition_utility_object.set_acq_func_type(acq_type)
                best_solution_found = kernel_optimiser_obj.optimise_kernel(i + 1)
                PH.printme(PH.p2, "Best*", i+1," \t", kernel_optimiser_obj.best_solution)

                maximum_accuracy = svm_wrapper_obj.compute_accuracy('HYPER', best_solution_found['best_kernel'],
                                                                                  hyper_gaussian_object)
                PH.printme(PH.p2, "Max:", maximum_accuracy)
                tot_max_acc = np.append(tot_max_acc, maximum_accuracy)
                tot_best_sol = np.append(tot_best_sol, best_solution_found)
                PH.printme(PH.p1, "\n\n*********************KF******R", i+1, "comp**********************\n\n\n\n")

        mean_max_acc = np.mean(tot_max_acc)
        PH.printme(PH.p1, "\n\n*******************MeanMaxAccuracy:", mean_max_acc,"*****************************")
        return mean_max_acc


if __name__ == "__main__":

    PH(os.getcwd())
    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")

    input = None

    argv = sys.argv[1:]
    cmd_inputs = {}

    try:
        opts, args = getopt.getopt(argv, "d:s:t:r:", ["dataset=", "subspaces=", "iterations=", "runs="])
    except getopt.GetoptError:
        print('python KerOptWrapper.py -d <dataset> -s <number_of_subspaces> -t <number_of_iterations> -r <runs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            cmd_inputs["dataset"] = arg
        elif opt in ("-s", "--subspaces"):
            cmd_inputs["subspaces"] = int(arg)
        elif opt in ("-t", "--iterations"):
            cmd_inputs["iterations"] = int(arg)
        elif opt in ("-r", "--runs"):
            cmd_inputs["runs"] = int(arg)
        else:
            print('python KerOptWrapper.py -d <dataset> -s <number_of_subspaces> -t <number_of_iterations> -r <runs>')
            sys.exit()

    ker_opt_wrapper_obj = KernelOptimizationWrapper()
    ker_opt_wrapper_obj.kernel_wrapper(stamp, input, cmd_inputs)

