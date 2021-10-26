import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
sys.path.append("..")
from HelperUtility.PrintHelper import PrintHelper as PH


# Base Class for running Bayesian Optmization Core
class BayesianOptimization:

    # Initialize the parameters of Bayesian Optimization Object
    def __init__(self, name, gp_object, acq_func_obj, func_helper_obj, no_iterations):

        self.name = name
        self.gp_obj = gp_object
        self.acq_func_obj = acq_func_obj
        self.no_iterations = no_iterations
        self.func_helper_obj = func_helper_obj
        # self.y_true_max = func_helper_obj.get_true_max()
        self.y_true_max = None



    def func(self,x):

        sum =0
        for item in x:
            sum += item
        value = np.exp(-x[0]) * np.sin(10*np.pi*x[0])
        return value

    def run_bayes_opt(self, run_count):

        # print the parameters for the current run
        PH.printme(PH.p3, "************ BO Run: " + str(
            run_count) + "     ACQ:" + self.acq_func_obj.acq_type.upper() + "    ***************\n\n")
        PH.printme(PH.p3, 'Initial Values for this run\n---X---\n', self.gp_obj.X)
        PH.printme(PH.p3, '---y---\n', self.gp_obj.y, '\n\n')

        # generate the points Xstar for the function evaluations
        random_points = []
        Xs = []

        # Generate specified (number of unseen data points) random numbers for each dimension
        for dim in np.arange(self.gp_obj.number_of_dimensions):
            random_data_point_each_dim = np.linspace(self.gp_obj.bounds[dim][0], self.gp_obj.bounds[dim][1],
                                                     self.gp_obj.number_of_test_datapoints).reshape(1,
                                                                                                    self.gp_obj.number_of_test_datapoints)
            random_points.append(random_data_point_each_dim)
        random_points = np.vstack(random_points)

        # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
        for sample_num in np.arange(self.gp_obj.number_of_test_datapoints):
            array = []
            for dim_count in np.arange(self.gp_obj.number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            Xs.append(array)
        Xs = np.vstack(Xs)

        # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently

        # #Replacing the code with SVM linkage
        # ys = self.func_helper_obj.get_true_func_value(Xs)

        ys=np.array([])

        #Commenting to avoid expensive evaluations here
        # for each_hyper_set in Xs:
        #     accuracy = self.func_helper_obj.get_true_func_value(each_hyper_set)
        #     ys=np.append(ys,accuracy)

        ###Normalizing code
        Xs = np.divide((Xs - self.gp_obj.Xmin), (self.gp_obj.Xmax - self.gp_obj.Xmin))
        # #normalizing y
        # Commenting to avoid expensive evaluations here
        # ys = (ys - self.gp_obj.ymin) / (self.gp_obj.ymax - self.gp_obj.ymin)

        # # # plot visuals before optimization, used when debugging each posterior distr in the 1D case
        # print ("Before Optimization" )
        # mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
        # standard_deviation = np.sqrt(diag_variance)
        # # # self.gp_obj.plot_posterior_predictions(10, Xs, ys, mean, standard_deviation)
        # self.gp_obj.plot_posterior_predictions('_' + self.gp_obj.kernel_char +'_' , Xs, ys, mean, standard_deviation)

        # Boolean to keep track of the function evaluations at X =[0], so that genuine vales are being added
        zero_value_bool = False
        regret = [[]]
        PH.printme(PH.p3, "Starting Optimization")

        # Run the optimization for the number of iterations and hence finding the best points to observe the function
        for i in range(self.no_iterations):
            PH.printme(PH.p2, "************###### BO"+ str(run_count)+ "i", i + 1,"_" ,self.gp_obj.kernel_char, " ######******************")

            ## Optimizing the characteristic length scale
            if self.gp_obj.len_scale_estimation and (i+1) % 3 == 0:

                # Estimating Length scale itself
                x_max_value = None
                log_like_max = - 1* float("inf")

                # Data structure to create the starting points for the scipy.minimize method
                random_points = []
                starting_points = []

                # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
                for dim in np.arange(self.gp_obj.number_of_dimensions):
                    random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_bounds[dim][0],
                                                                   self.gp_obj.len_scale_bounds[dim][1],
                                                                   self.gp_obj.number_of_restarts_likelihood).\
                                                                    reshape(1,self.gp_obj.number_of_restarts_likelihood)
                    random_points.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points = np.vstack(random_points)

                # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
                for sample_num in np.arange(self.gp_obj.number_of_restarts_likelihood):
                    array = []
                    for dim_count in np.arange(self.gp_obj.number_of_dimensions):
                        array.append(random_points[dim_count, sample_num])
                    starting_points.append(array)
                starting_points = np.vstack(starting_points)

                variance_start_points = np.random.uniform(self.gp_obj.signal_variance_bounds[0],
                                                                   self.gp_obj.signal_variance_bounds[1],
                                                                   self.gp_obj.number_of_restarts_likelihood)

                total_bounds = self.gp_obj.len_scale_bounds.copy()
                total_bounds.append(self.gp_obj.signal_variance_bounds)

                for ind in np.arange(self.gp_obj.number_of_restarts_likelihood):

                    init_len_scale = starting_points[ind]
                    init_var = variance_start_points[ind]

                    init_points = np.append(init_len_scale, init_var)
                    maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood_l(x),
                                          init_points,
                                          method='L-BFGS-B',
                                          tol=0.01,
                                          options={'maxfun': 20, 'maxiter': 20},
                                          bounds=total_bounds)

                    len_scale_temp = maxima['x'][:self.gp_obj.number_of_dimensions]
                    variance_temp = maxima['x'][len(maxima['x']) - 1]
                    params = np.append (len_scale_temp, variance_temp)
                    log_likelihood = self.gp_obj.optimize_log_marginal_likelihood_l(params)

                    if (log_likelihood > log_like_max ):
                        PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for l= ",
                              maxima['x'][: self.gp_obj.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])
                        x_max_value = maxima
                        log_like_max = log_likelihood

                self.gp_obj.char_length_scale = x_max_value['x'][:self.gp_obj.number_of_dimensions]
                self.gp_obj.signal_variance = x_max_value['x'][len(maxima['x']) - 1]

                PH.printme(PH.p1, "Opt Length scale: ", self.gp_obj.char_length_scale, "\nOpt variance: ", self.gp_obj.signal_variance)
                self.gp_obj.L_x_x = self.gp_obj.compute_l(self.gp_obj.X)

            ## Optimizing the characteristic length scale parameters : ***** working for 1D only
            if (self.gp_obj.weights_estimation and (i + 1) % 3 == 0):

                x_max_value = None
                log_like_max = - 1 * float("inf")

                random_points_a = []
                random_points_b = []
                random_points_c = []
                random_points_d = []

                self.gp_obj.number_of_restarts_likelihood = 5

                # Data structure to create the starting points for the scipy.minimize method
                random_data_point_each_dim = np.random.uniform(self.gp_obj.len_weights_bounds[0][0],
                                                               self.gp_obj.len_weights_bounds[0][1],
                                                               self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                                                                  self.gp_obj.number_of_restarts_likelihood)
                random_points_a.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.gp_obj.len_weights_bounds[1][0],
                                                               self.gp_obj.len_weights_bounds[1][1],
                                                               self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                                                                  self.gp_obj.number_of_restarts_likelihood)
                random_points_b.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.gp_obj.len_weights_bounds[2][0],
                                                               self.gp_obj.len_weights_bounds[2][1],
                                                               self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                                                                  self.gp_obj.number_of_restarts_likelihood)
                random_points_c.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.gp_obj.len_weights_bounds[3][0],
                                                               self.gp_obj.len_weights_bounds[3][1],
                                                               self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                                                                  self.gp_obj.number_of_restarts_likelihood)
                random_points_d.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points_a = np.vstack(random_points_a)
                random_points_b = np.vstack(random_points_b)
                random_points_c = np.vstack(random_points_c)
                random_points_d = np.vstack(random_points_d)

                variance_start_points = np.random.uniform(self.gp_obj.signal_variance_bounds[0],
                                                          self.gp_obj.signal_variance_bounds[1],
                                                          self.gp_obj.number_of_restarts_likelihood)

                for ind in np.arange(self.gp_obj.number_of_restarts_likelihood):

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
                    total_bounds = self.gp_obj.len_weights_bounds.copy()
                    total_bounds.append(self.gp_obj.signal_variance_bounds)

                    maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood_weight_params(x),
                                          tot_init_points,
                                          method='L-BFGS-B',
                                          tol=0.01,
                                          options={'maxfun': 20, 'maxiter': 20},
                                          bounds=total_bounds)

                    params = maxima['x']
                    log_likelihood = self.gp_obj.optimize_log_marginal_likelihood_weight_params(params)
                    if (log_likelihood > log_like_max):
                        PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for params ", params)
                        x_max_value = maxima['x']
                        log_like_max = log_likelihood

                if not self.gp_obj.disp_bool:
                    PH.printme(PH.p4, "!!!!!!!!!!!Matrix went non positive definite, inverting with pinv")
                self.disp_bool = True

                self.gp_obj.len_weights = x_max_value[0:4]
                self.gp_obj.signal_variance = x_max_value[len(maxima['x']) - 1]

                PH.printme(PH.p1, "Opt weights: ", self.gp_obj.len_weights, "   variance:", self.gp_obj.signal_variance)

                # Recomputing L according to the updated length scale
                self.gp_obj.L_x_x = self.gp_obj.compute_l(self.gp_obj.X)

            ## Optimizing the characteristic length scale parameters : ***** working for 1D only
            if (self.gp_obj.params_estimation and (i + 1) % 3 == 0):

                x_max_value = None
                log_like_max = - 1* float("inf")


                random_points_a = []
                random_points_b = []
                random_points_c = []

                for d in np.arange(self.gp_obj.number_of_dimensions):
                    # Data structure to create the starting points for the scipy.minimize method
                    random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[d][0][0],
                                                                   self.gp_obj.len_scale_param_bounds[d][0][1],
                                                                   self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                   self.gp_obj.number_of_restarts_likelihood)
                    random_points_a.append(random_data_point_each_dim)


                    random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[d][1][0],
                                                                   self.gp_obj.len_scale_param_bounds[d][1][1],
                                                                   self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                   self.gp_obj.number_of_restarts_likelihood)
                    random_points_b.append(random_data_point_each_dim)

                    if self.gp_obj.len_scale_func_type[d] == 'linear' or self.gp_obj.len_scale_func_type[d] == 'gaussian' :
                        # Two Parameters by default
                        pass

                    elif self.gp_obj.len_scale_func_type[d] == 'quadratic' :

                        random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[d][2][0],
                                                                       self.gp_obj.len_scale_param_bounds[d][2][1],
                                                                       self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                       self.gp_obj.number_of_restarts_likelihood)
                        random_points_c.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points_a = np.vstack(random_points_a)
                random_points_b = np.vstack(random_points_b)

                variance_start_points = np.random.uniform(self.gp_obj.signal_variance_bounds[0],
                                                          self.gp_obj.signal_variance_bounds[1],
                                                          self.gp_obj.number_of_restarts_likelihood)

                if 'quadratic' in self.gp_obj.len_scale_func_type:
                    random_points_c = np.vstack(random_points_c)

                for ind in np.arange(self.gp_obj.number_of_restarts_likelihood):

                    tot_init_points = []
                    total_bounds = []
                    count=0
                    for each_dimension in np.arange(self.gp_obj.number_of_dimensions):
                        param_a = random_points_a[each_dimension][ind]
                        tot_init_points.append(param_a)
                        param_b = random_points_b[each_dimension][ind]
                        tot_init_points.append(param_b)
                        total_bounds.append(self.gp_obj.len_scale_param_bounds[each_dimension][0])
                        total_bounds.append(self.gp_obj.len_scale_param_bounds[each_dimension][1])

                        if self.gp_obj.len_scale_func_type[each_dimension] == "linear" \
                                or self.gp_obj.len_scale_func_type[each_dimension]  == "gaussian":
                            pass

                        elif self.gp_obj.len_scale_func_type[each_dimension] == "quadratic":
                            param_c = random_points_c[count][ind]
                            tot_init_points.append(param_c)
                            count= count+1
                            total_bounds.append(self.gp_obj.len_scale_param_bounds[each_dimension][2])
                            pass

                    tot_init_points.append(variance_start_points[ind])
                    total_bounds.append(self.gp_obj.signal_variance_bounds)

                    maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood_l_params(x),
                                              tot_init_points,
                                              method='L-BFGS-B',
                                              tol=0.01,
                                              options={'maxfun': 20, 'maxiter': 20},
                                              bounds=total_bounds)

                    params = maxima['x']
                    log_likelihood = self.gp_obj.optimize_log_marginal_likelihood_l_params(params)
                    if (log_likelihood > log_like_max):
                        PH.printme(PH.p1, ind," New maximum log likelihood ", log_likelihood, " found for params ", params)
                        x_max_value = maxima['x']
                        log_like_max = log_likelihood
                total_len_scale_params = []
                param_count = 0
                for func_type in self.gp_obj.len_scale_func_type:
                    if(func_type == 'linear') or (func_type == 'gaussian'):
                        total_len_scale_params.append(x_max_value[param_count:(param_count+2)])
                        param_count += 2
                    elif(func_type == 'quadratic'):
                        total_len_scale_params.append(x_max_value[param_count:(param_count+3)])
                        param_count += 3
                self.gp_obj.len_scale_params = total_len_scale_params
                self.gp_obj.signal_variance = x_max_value[len(maxima['x']) - 1]

                PH.printme(PH.p1, "Opt Params: ", self.gp_obj.len_scale_params, "   variance:", self.gp_obj.signal_variance)

                # Recomputing L according to the updated length scale
                self.gp_obj.L_x_x = self.gp_obj.compute_l(self.gp_obj.X)


            # Random search to find the maxima in the specified bounds
            if (self.acq_func_obj.acq_type == 'rs'):
                PH.printme(PH.p1, "performing RS for the maxima")

                # Randomly select the values for Xnew and perform random search for optima in the search space
                xnew = np.array([])
                for dim in np.arange(self.gp_obj.number_of_dimensions):
                    value = np.random.uniform(self.gp_obj.bounds[dim][0], self.gp_obj.bounds[dim][1], 1).reshape(1, 1)
                    xnew = np.append(xnew, value)

            else:
                # Maximising the acquisition function to obtain the value at which the function has to be evaluated next
                xnew, acq_func_values = self.acq_func_obj.max_acq_func(self.gp_obj, Xs, ys, i + 1)


            # Notify if zeroes are being added as the best point to evaluate next
            if (np.array_equal(xnew, np.zeros(self.gp_obj.number_of_dimensions))):
                zero_value_bool = True
                PH.printme(PH.p4, '\nzeroes encountered in ', run_count, " iteration: ", i + 1)

            xnew_orig = np.multiply(xnew.T, (self.gp_obj.Xmax - self.gp_obj.Xmin)) + self.gp_obj.Xmin
            # #xnew_orig = xnew * (self.gp_obj.Xmax - self.gp_obj.Xmin) + self.gp_obj.Xmin

            PH.printme(PH.p2, "xnew:", xnew, "\noriginal",)

            ynew_orig = self.func_helper_obj.get_true_func_value(xnew_orig)

            # Add the new observation point to the existing set of observed samples along with its true value
            X = self.gp_obj.X
            X = np.append(X, [xnew], axis=0)

            # calculate the true function value and add it to the existing set of values
            # # # Normalizing
            ynew = (ynew_orig - self.gp_obj.ymin)/ (self.gp_obj.ymax - self.gp_obj.ymin)

            # Commenting ynew augment for standardizing logic implemented above
            y = np.append(self.gp_obj.y, [ynew], axis=0)
            PH.printme(PH.p2, "(",xnew,ynew, ") is the new value added..    Original: ", (xnew_orig, ynew_orig) )

            # Refit the GP model to use the updated prior knowledge
            self.gp_obj.gaussian_fit(X, y)

            regret = np.append(regret, y.max())
            PH.printme(PH.p3, "\n")

        # Display the final values for this iteration
        PH.printme(PH.p3, 'Final values:\n$X: \n', self.gp_obj.X.T, '\n$y:\n', self.gp_obj.y.T)
        PH.printme(PH.p3, "True Max: ", self.y_true_max)
        PH.printme(PH.p3, self.acq_func_obj.acq_type.upper(), self.gp_obj.kernel_char,", Observed Maximum value: ", self.gp_obj.y.max())
        PH.printme(PH.p3, "Regret: ", regret)
        PH.printme(PH.p3, "\n\n\n")

        return regret

    def plot_regret(self, regret, iterations):

        # Plot the regret for each of the iteration
        iterations_axes = np.arange(start=1, stop=iterations + 1, step=1)
        plt.figure("Regret ")
        plt.clf()
        plt.plot(iterations_axes, regret, 'b')
        plt.axis([1, iterations, 0, 1])
        plt.title('Regret for iterations: ' + str(iterations))
        plt.savefig('regret.png')

