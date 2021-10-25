import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH


class KernelOptimiser:

    def __init__(self, hyper_gaussian_object, acquisition_utility_object, number_of_subspace_selection_iterations,
                 number_of_iterations_best_solution, number_of_init_random_kernel_y_observations, svm_wrapper_obj):
        self.hyper_gaussian_object = hyper_gaussian_object
        self.acquisition_utility_object = acquisition_utility_object
        self.number_of_subspace_selection_iterations = number_of_subspace_selection_iterations
        self.number_of_iterations_best_solution = number_of_iterations_best_solution
        self.number_of_init_random_kernel_y_observations = number_of_init_random_kernel_y_observations
        self.best_solution = {}
        self.svm_wrapper_obj = svm_wrapper_obj
        #
        self.complete_dataset_kernel_observations = np.array([]).reshape(-1, self.hyper_gaussian_object.no_principal_components)
        self.complete_dataset_y = np.array([]).reshape(-1, 1)

    def generate_observations(self, kernel_bias, basis_weights, kernel_samples):

        # Generate initial observations for the kernel optimisation with the selected basis

        observations_kernel =  kernel_bias + basis_weights[0] * kernel_samples[0]

        # Process for running observations
        observations_y = self.svm_wrapper_obj.compute_accuracy('HYPER', observations_kernel, self.hyper_gaussian_object)
        return observations_kernel, observations_y

    def sample_basis_weights(self):
        # Generate weights to the selected basis vectors
        weights = np.random.uniform(self.hyper_gaussian_object.basis_weights_bounds[0], self.hyper_gaussian_object.basis_weights_bounds[
            1], self.hyper_gaussian_object.number_of_basis_vectors_chosen).reshape(-1, 1)

        return weights

    def optimise_kernel(self, run_count):

        # print the parameters for the current run
        PH.printme(PH.p2, "***************KF" + str(run_count) + "***************\n\n")
        PH.printme(PH.p2, 'Initial Grid Values selected for this run\n--- X in Grid ---\n', self.hyper_gaussian_object.X)

        # Without kappa EVD
        # self.best_solution['best_kernel'] = np.zeros(shape=(np.square(self.hyper_gaussian_object.number_of_samples_in_X_for_grid),
        #                                                     1)).reshape(1, -1)

        # # Eigen based best solution
        self.best_solution['best_kernel'] = np.zeros(shape=(1, self.hyper_gaussian_object.no_principal_components))
        PH.printme(PH.p1, "Starting Optimization")

        bool_comp = False

        # Run the optimization for the number of iterations and hence finding the best points to observe the function
        self.hyper_gaussian_object.compute_kappa_utils()

        for subspace_selection_count in range(self.number_of_subspace_selection_iterations):

            PH.printme(PH.p1, "@@@@@@@@@@@@@@KFO@@@@@@@@@@@@@@R" + str(run_count) + "s", subspace_selection_count + 1,
                       "@@@@@@@@@@@@@@@@@@@@@@@@KFO@@@@@@@@@@@@@@@@@")
            self.hyper_gaussian_object.current_kernel_samples = self.hyper_gaussian_object.generate_basis_as_kernel_samples()
            self.hyper_gaussian_object.current_kernel_bias = self.best_solution['best_kernel']

            for number_init_random_kernel_y_obs in range(self.number_of_init_random_kernel_y_observations):
                basis_weights = self.sample_basis_weights()
                observations_kernel, observations_y = self.generate_observations(self.hyper_gaussian_object.current_kernel_bias,
                                                                                 basis_weights,
                                                                                 self.hyper_gaussian_object.current_kernel_samples)

                self.hyper_gaussian_object.observations_kernel.append(observations_kernel)
                self.hyper_gaussian_object.observations_y = np.append(self.hyper_gaussian_object.observations_y, observations_y)
                PH.printme(PH.p2, "Generated new observation with observation_y: ", observations_y)

            # Vertical stacking for making it suitable for computation
            self.hyper_gaussian_object.observations_kernel = np.vstack(self.hyper_gaussian_object.observations_kernel)
            self.hyper_gaussian_object.observations_y = np.vstack(self.hyper_gaussian_object.observations_y)

            # Search for the best solution in the subspace selected
            for best_solution_count in range(self.number_of_iterations_best_solution):
                PH.printme(PH.p2, "\n###########################besti", best_solution_count+1, "s",
                           (subspace_selection_count+1),"####################################")
                PH.printme(PH.p1, "Using log marginal likelihood to compute the optimised length scale for the kernels")
                self.hyper_gaussian_object.compute_hyperparams_kernel_observations()
                PH.printme(PH.p1, "optimised values: l = ", self.hyper_gaussian_object.char_length_scale, "  sig = ",
                      self.hyper_gaussian_object.signal_variance)

                PH.printme(PH.p1, "pre-calculating COV matrix for kernel observations")
                cov_K_K_hypergp = self.hyper_gaussian_object.compute_covariance_matrix_for_kernels(
                    self.hyper_gaussian_object.observations_kernel, self.hyper_gaussian_object.observations_kernel)
                self.hyper_gaussian_object.L_K_K_hypergp = np.linalg.cholesky(cov_K_K_hypergp + 1e-6 * np.eye(len(
                    self.hyper_gaussian_object.observations_kernel)))

                best_basis_weights = self.acquisition_utility_object.maximise_acq_function(self.hyper_gaussian_object,
                                                                                           best_solution_count+1)
                PH.printme(PH.p2, "Best weights: ", best_basis_weights)
                observations_kernel_new, observations_y_new = self.generate_observations(self.hyper_gaussian_object.current_kernel_bias,
                                                                                         best_basis_weights,
                                                                                         self.hyper_gaussian_object.current_kernel_samples)
                PH.printme(PH.p1, "New observation found in the given subspace")
                PH.printme(PH.p1, "New kernel suggested: ", observations_kernel_new)
                PH.printme(PH.p2, "with observation value: ", observations_y_new)
                self.hyper_gaussian_object.observations_y = np.append(self.hyper_gaussian_object.observations_y, [observations_y_new[0]],
                                                                      axis=0)
                self.hyper_gaussian_object.observations_kernel = np.append(self.hyper_gaussian_object.observations_kernel,
                                                                           [observations_kernel_new[0]], axis=0)

                # Resetting the cached covariance matrix for kernel observations
                self.hyper_gaussian_object.L_K_K_hypergp = None

            PH.printme(PH.p1, "@@@@@@@@@@@@@@@@@@@@@@@@@s", (subspace_selection_count + 1), "comp@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            self.complete_dataset_y = np.append(self.complete_dataset_y, self.hyper_gaussian_object.observations_y, axis=0)
            self.complete_dataset_kernel_observations = np.append(self.complete_dataset_kernel_observations,
                                                                  self.hyper_gaussian_object.observations_kernel, axis=0)
            index_subspace_best_value = self.complete_dataset_y.argmax()
            self.best_solution['best_kernel'] = np.array(self.complete_dataset_kernel_observations[index_subspace_best_value]).reshape(1,
                                                                                                                                       -1)
            self.best_solution['best_value'] = self.complete_dataset_y[index_subspace_best_value]
            self.hyper_gaussian_object.observations_kernel = []
            self.hyper_gaussian_object.observations_y = []
            self.hyper_gaussian_object.best_kernel = self.best_solution['best_kernel']

            PH.printme(PH.p1, "Best solution is selected as \n", self.best_solution)
            PH.printme(PH.p1, "@@@@@@@@@@@@@@@@@@@@@@@@@@s", (subspace_selection_count+1), "comp@@@@@@@@@@@@@@@@@@@@\n\n\n\n")
            PH.printme(PH.p2, "Data_after:s", (subspace_selection_count+1))
            PH.printme(PH.p2, "Observation_y:", self.complete_dataset_y.T, "\n")

        return self.best_solution




