from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys
import scipy.optimize as opt
sys.path.append("../..")
np.random.seed(200)
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))


from sklearn.model_selection import GridSearchCV

class svm_var:

    def __init__(self):
        self.dataset_input_file = None
        self.X_train = None
        self.Xtest = None
        self.y_train = None
        self.y_test = None
        self.D = None
        self.f = None
        self.len_weights = []


    def compute_kernel_matrix(self, data_point1, data_point2, char_len_scale, signal_variance):

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
                each_kernel_val = self.len_weights[0] * sek + self.len_weights[1] * mat3 + self.len_weights[2] * lin
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat


    def optimise_weights(self):

        x_max_value = None
        log_like_max = - 1 * float("inf")

        random_points_a = []
        random_points_b = []
        random_points_c = []

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



        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points_a = np.vstack(random_points_a)
        random_points_b = np.vstack(random_points_b)
        random_points_c = np.vstack(random_points_c)
        variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                  self.signal_variance_bounds[1],
                                                  self.number_of_restarts_likelihood)

        for ind in np.arange(self.number_of_restarts_likelihood):

            # print("Restart: ", ind)
            tot_init_points = []

            param_a = random_points_a[0][ind]
            tot_init_points.append(param_a)
            param_b = random_points_b[0][ind]
            tot_init_points.append(param_b)
            param_c = random_points_c[0][ind]
            tot_init_points.append(param_c)
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
                print("New maximum log likelihood ", log_likelihood, " found for params ", params)
                x_max_value = maxima['x']
                log_like_max = log_likelihood

        self.len_weights = x_max_value[0:4]
        self.signal_variance = x_max_value[len(maxima['x']) - 1]

        print("Opt weights: ", self.len_weights, "   variance:", self.signal_variance)



    def optimize_log_marginal_likelihood_weight_params(self, input):

        # print("optimising with ", input)
        self.len_weights = input[0:3]
        # Following parameters not used in any computations
        signal_variance = input[len(input) - 1]
        self.signal_variance = signal_variance

        K_x_x = self.compute_kernel_matrix(self.X_train, self.X_train, self.char_len_scale, signal_variance)
        eye = 1e-16 * np.eye(len(self.X_train))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y_train)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood


    def compute_accuracy(self, variation):

        if variation == "nu":

            grid_values = {'nu': [0.01, 0.001, 0.4, 0.5]
                           # ,'gamma': [0.001, 0.01, 0.1, 1, 10, 1000, 10000]
                           }
            nusvc = NuSVC(kernel=kernel_type, gamma="auto", random_state=42)
            grid_nusvm_acc = GridSearchCV(nusvc, param_grid=grid_values, refit=True, verbose=1)
            print("Fitting SVM for the Data .... ")
            grid_nusvm_acc.fit(self.X_train, self.y_train)
            print("Predicting accuracy....")
            y_pred = grid_nusvm_acc.predict(self.X_test)

        elif variation == "c":
            grid_values = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                           }
            svc = SVC(random_state=42)
            grid_svm_acc = GridSearchCV(svc, param_grid=grid_values, refit=True, verbose=1)
            print("Fitting SVM for the Data .... ")
            grid_svm_acc.fit(self.X_train, self.y_train)
            print("Predicting Values...")
            y_pred = grid_svm_acc.predict(self.X_test)

        elif variation == "rbf":
            svc = SVC(C=1, kernel=kernel_type, random_state=42)
            print("Fitting SVM for the Data .... ")
            svc.fit(self.X_train, self.y_train)
            print("Predicting Values...")
            y_pred = svc.predict(self.X_test)

        elif variation == "lag":
            grid_values = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                           }
            svc = SVC(kernel = kernel_type, random_state=42)
            grid_svm_acc = GridSearchCV(svc, param_grid=grid_values, refit=True, verbose=1)
            print("Fitting SVM for the Data .... ")
            grid_svm_acc.fit(self.X_train, self.y_train)
            print("Predicting Values...")
            y_pred = grid_svm_acc.predict(self.X_test)


        elif variation == "multi":
            grid_values = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                           }

            self.len_weights_bounds = [[0.1, 3] for i in range(3)]
            self.len_weights = [0.4, 0.3, 0.3]
            self.signal_variance_bounds = [0.1, 1]
            self.number_of_restarts_likelihood = 50
            self.number_of_observed_samples = self.X_train.shape[0]

            self.char_len_scale = 0.4
            self.optimise_weights()
            print(self.len_weights)

            svc = SVC(kernel=kernel_type, random_state=42)
            grid_svm_acc = GridSearchCV(svc, param_grid=grid_values, refit=True, verbose=1)
            print(svc)
            print("Computing Xtr_Xtr .... ")
            kernel_mat_Xtr_Xtr = self.compute_kernel_matrix(self.X_train, self.X_train, self.char_len_scale, self.signal_variance)
            print("Fitting SVM for the Data .... ")
            grid_svm_acc.fit(kernel_mat_Xtr_Xtr, self.y_train)
            print("Computing Xte_Xtr .... ")
            kernel_mat_Xte_Xtr = self.compute_kernel_matrix(self.X_test, self.X_train, self.char_len_scale, self.signal_variance)
            print("Predicting Values...")
            y_pred = grid_svm_acc.predict(kernel_mat_Xte_Xtr)
            accuracy = accuracy_score(self.y_test, y_pred)
            print("Accuracy: ", accuracy)
            return np.array([[accuracy]])

        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy: ", accuracy)
        return accuracy

    def configure_svms(self, dataset, kernel_type, variation):

        if dataset == "ECO":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_ecoli.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "SEED":
            self.dataset_input_file = "../DatasetUtils/Dataset/Seed_Data.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "PIMA":
            self.dataset_input_file = "../DatasetUtils/Dataset/diabetes.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "DERMATOLOGY":
            self.dataset_input_file = "../DatasetUtils/Dataset/dermatology.csv"
            total_dataframe = pd.read_csv(self.dataset_input_file, na_values='?')
            dataframe = total_dataframe.fillna(method='bfill')

            f = dataframe.iloc[:, -1]
            dataframe = dataframe.drop(dataframe.columns[-1], axis=1)
            D = dataframe

            dataframe_cat = dataframe.select_dtypes(include=[np.number])
            label_encoder = preprocessing.LabelEncoder()
            labeled_dataframe_cat = dataframe_cat.apply(label_encoder.fit_transform)
            #
            onehot_encoder = preprocessing.OneHotEncoder()
            onehot_encoder.fit(labeled_dataframe_cat)
            onehotlabel_cols = onehot_encoder.transform(labeled_dataframe_cat).toarray()
            D_std = onehotlabel_cols

        elif dataset == "WINE":
            self.dataset_input_file = "../DatasetUtils/Dataset/Wine.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "BIO":
            self.dataset_input_file = "../DatasetUtils/Dataset/data_biodeg.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "CREDIT":
            self.dataset_input_file = "../DatasetUtils/Dataset/credit.csv"
            total_dataframe = pd.read_csv(self.dataset_input_file, na_values='?')
            dataframe = total_dataframe.fillna(method='bfill')

            f = dataframe.iloc[:, -1]
            dataframe = dataframe.drop(dataframe.columns[-1], axis=1)
            D = dataframe

            dataframe_num = dataframe.select_dtypes(include=[np.number])
            min_max_scaler = MinMaxScaler()
            dataframe_num_std = min_max_scaler.fit_transform(dataframe_num)

            dataframe_cat = dataframe.select_dtypes(include=[object])
            label_encoder = preprocessing.LabelEncoder()
            labeled_dataframe_cat = dataframe_cat.apply(label_encoder.fit_transform)
            #
            onehot_encoder = preprocessing.OneHotEncoder()
            onehot_encoder.fit(labeled_dataframe_cat)
            onehotlabel_cols = onehot_encoder.transform(labeled_dataframe_cat).toarray()

            D_std = np.concatenate((dataframe_num_std, onehotlabel_cols), axis=1)

        elif dataset == "CONTRA":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_contra.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "PHO":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_phoneme.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "HAY":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_hayes.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "ECO":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_ecoli.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "CAR":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_car.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        # Store and Split the dataset
        self.D = D
        self.f = f
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(D_std, f, test_size=0.20)
        accuracy = self.compute_accuracy(variation)
        return accuracy


if __name__ == "__main__":

    svm_var_obj = svm_var()
    runs = 10
    # variation = "nu"
    variation = "c"
    # variation = "rbf"
    # variation = "lag"

    # MKL Addition
    # variation = "multi"

    kernel_type = "rbf"
    # kernel_type = "poly"

    # MKL Addition
    kernel_type = "precomputed"

    # dataset = "CAR"
    # dataset = "ECO"
    # dataset = "SEED"
    # dataset = "PIMA"
    # dataset = "DERMATOLOGY"
    dataset = "WINE"
    # dataset = "BIO"
    # dataset = "CONTRA"
    # dataset = "PHO"
    # dataset = "HAY"
    # dataset = "CREDIT"
    # dataset = "CAR"

    print("Dataset: ", dataset)
    acc_list = []
    for i in np.arange(runs):
        acc = svm_var_obj.configure_svms(dataset, kernel_type, variation)
        acc_list.append(acc)
    print(acc_list)
    print("Mean: ",np.mean(acc_list), "\tStd:", np.std(acc_list))
