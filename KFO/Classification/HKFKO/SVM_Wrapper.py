import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV


import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH


class SVM_Wrapper:

    def __init__(self):
        self.dataset_input_file = None
        self.X_train = None
        self.Xtest = None
        self.y_train = None
        self.y_test = None
        self.D = None
        self.f = None

    def construct_svm_classifier(self, dataset):

        if dataset == "wdbc":
            self.dataset_input_file = "../DatasetUtils/Dataset/wdbc.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            D = total_data.drop(total_data.columns[0], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

            f = total_data.iloc[:, 0]

        elif dataset == "iris":
            self.dataset_input_file = "../DatasetUtils/Dataset/iris.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "glass":
            self.dataset_input_file = "../DatasetUtils/Dataset/glass.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "ionos":
            self.dataset_input_file = "../DatasetUtils/Dataset/ionosphere.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "sonar":
            self.dataset_input_file = "../DatasetUtils/Dataset/sonar.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "heart":
            self.dataset_input_file = "../DatasetUtils/Dataset/heart.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "credit":
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

        elif dataset == "credit_arc":
            self.dataset_input_file = "../DatasetUtils/Dataset/German_Credit.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "seeds":

            self.dataset_input_file = "../DatasetUtils/Dataset/Seed_Data.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "pima":
            self.dataset_input_file = "../DatasetUtils/Dataset/diabetes.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "dermatology":
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

        elif dataset == "wine":
            self.dataset_input_file = "../DatasetUtils/Dataset/Wine.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "bio":
            self.dataset_input_file = "../DatasetUtils/Dataset/data_biodeg.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "contra":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_contra.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "pho":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_phoneme.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "hay":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_hayes.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "eco":
            self.dataset_input_file = "../DatasetUtils/Dataset/dataset_ecoli.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "car":
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



    def compute_kerenel_mat_hyperk(self, data_point1, data_point2, observations_kernel, hypergp_obj):
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                dp1 = data_point1[i]
                dp2 = data_point2[j]
                each_kernel_val = hypergp_obj.estimate_kernel_for_Xtil(dp1, dp2, observations_kernel)

                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def normalise_numerical_columns(self, D, cols):

        for each_col in cols:
            max_val = max(D.iloc[:, each_col])
            min_val = min(D.iloc[:, each_col])
            D.iloc[:, each_col] = (D.iloc[:, each_col] - min_val)/(max_val - min_val)
        return D

    def compute_accuracy(self, kernel_type, observations_kernel, hyperGP_obj):

        print("\nComputing accuracy for the kernel... ", observations_kernel[0][:4],"\nConstructing SVM Classifier")

        # Grid search
        grid_values = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                       }
        svc = SVC(kernel='precomputed', random_state=42)
        grid_svm_acc = GridSearchCV(svc, param_grid=grid_values, refit=True, verbose=1)
        PH.printme(PH.p1, "Computing Xtr_Xtr .... ")
        kernel_mat_Xtr_Xtr = self.compute_kerenel_mat_hyperk(self.X_train, self.X_train, observations_kernel, hyperGP_obj)
        PH.printme(PH.p1, "Clipping Training Matrix")
        kernel_mat_eigen_values, kernel_mat_eigen_vectors = np.linalg.eigh(kernel_mat_Xtr_Xtr)

        # Flip
        # eig_sign = np.sign(kernel_mat_eigen_values)
        # updated_eigen_diag = np.diag(eig_sign)

        # Clip
        # kernel_mat_eigen_values[kernel_mat_eigen_values < 0] = 0
        # updated_eigen_diag = np.diag(kernel_mat_eigen_values)

        # Clip Indicator Function
        kernel_mat_eigen_values[kernel_mat_eigen_values < 0] = 0
        kernel_mat_eigen_values[kernel_mat_eigen_values >= 0] = 1
        updated_eigen_diag = np.diag(kernel_mat_eigen_values)

        # kernel_mat_Xtr_Xtr = np.dot(np.dot(kernel_mat_eigen_vectors, (np.dot(updated_eigen_diag, kernel_mat_eigen_vectors.T))),
        #                             kernel_mat_Xtr_Xtr)
        PH.printme(PH.p1, "Clipping Train Done...")
        PH.printme(PH.p1, "Fitting SVM for the Data .... ")
        grid_svm_acc.fit(kernel_mat_Xtr_Xtr, self.y_train)
        PH.printme(PH.p1, "Computing Xte_Xtr .... ")
        kernel_mat_Xte_Xtr = self.compute_kerenel_mat_hyperk(self.X_test, self.X_train, observations_kernel, hyperGP_obj)

        PH.printme(PH.p1, "Clipping Test Matrix")
        kernel_mat_Xte_Xtr = np.dot(kernel_mat_Xte_Xtr, np.dot(kernel_mat_eigen_vectors, (np.dot(updated_eigen_diag,
                                                                                                kernel_mat_eigen_vectors.T))))
        PH.printme(PH.p1, "Clipping Test Done...")
        PH.printme(PH.p1, "Predicting Values...")
        y_pred = grid_svm_acc.predict(kernel_mat_Xte_Xtr)
        accuracy = accuracy_score(self.y_test, y_pred)
        PH.printme(PH.p1, "Accuracy: ", accuracy)
        return np.array([[accuracy]])

