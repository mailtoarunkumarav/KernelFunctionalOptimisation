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

from MKLpy import generators
from MKLpy.preprocessing import kernel_normalization
from MKLpy.model_selection import train_test_split
from MKLpy.algorithms import EasyMKL
from itertools import product
from MKLpy.algorithms import EasyMKL
from MKLpy.model_selection import cross_val_score
from MKLpy.metrics import pairwise

from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

class svm_mkl:

    def __init__(self):
        self.dataset_input_file = None
        self.X_train = None
        self.Xtest = None
        self.y_train = None
        self.y_test = None
        self.D = None
        self.f = None
        self.len_weights = []


    def configure_svms(self, dataset):

        if dataset == "SEED":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\Seed_Data.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "WINE":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\Wine.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "BIO":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\data_biodeg.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "CREDIT":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\credit.csv"
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
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\dataset_contra.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "PHO":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\dataset_phoneme.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "HAY":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\dataset_hayes.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "ECO":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\dataset_ecoli.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        elif dataset == "CAR":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\dataset_car.csv"

            total_data = pd.read_csv(self.dataset_input_file)
            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        if dataset == "WDBC":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\wdbc.csv"
            total_data = pd.read_csv(self.dataset_input_file)
            D = total_data.drop(total_data.columns[0], axis=1)

            #Method 1
            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

            # Method2
            # cols = np.arange(0, len(D.iloc[0]))
            # D_std = self.normalise_numerical_columns(D, cols)

            f = total_data.iloc[:, 0]

        elif dataset == "IRIS":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\iris.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "GLASS":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\glass.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "IONOS":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\ionosphere.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)
            # print(D, "\n\n", D_std)
            # exit(0)

        elif dataset == "SONAR":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\sonar.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)

        elif dataset == "HEART":
            self.dataset_input_file = "..\..\DatasetUtils\Dataset\heart.csv"
            total_data = pd.read_csv(self.dataset_input_file)

            f = total_data.iloc[:, -1]
            D = total_data.drop(total_data.columns[-1], axis=1)

            min_max_scaler = MinMaxScaler()
            D_std = min_max_scaler.fit_transform(D)


        # # Sample IRIS data
        # from sklearn.datasets import load_breast_cancer
        # ds = load_breast_cancer()
        # X, Y = ds.data, ds.target
        # print(type(Y))

        f = np.array(f.tolist())

        # 2 custom kernels (linear and polynomial)
        # K_mix = generators.RBF_generator(D_std, gamma=[.001])

        # ker_functions = [rbf_kernel, linear_kernel, polynomial_kernel]
        # ker_functions = [rbf_kernel, linear_kernel]
        ker_functions = [rbf_kernel]
        K_mix = generators.Lambda_generator(D_std, kernels=ker_functions)
        print(ker_functions)

        # Not used
        #KL_norm = [kernel_normalization(K) for K in K_mix]

        KLtr, KLte, Ytr, Yte = train_test_split(K_mix, f, test_size=.2, random_state=42)

        lam_values = [0, 0.1, 0.2, 0.3 , 0.5, 1]
        C_values = [0.01, 0.1 , 1, 10, 100]

        scores_list = np.array([])
        for lam, C in product(lam_values, C_values):
            svm = SVC(C=C)


            mkl = EasyMKL(lam=lam, learner=svm)
            scores = cross_val_score(K_mix, f, mkl, n_folds=3, scoring='accuracy')
            mean_score = (scores[0]+scores[1]+scores[2])/3
            print(lam, C, scores, mean_score)
            scores_list = np.append(scores_list, mean_score)
        return scores_list.max()



if __name__ == "__main__":

    svm_var_obj = svm_mkl()
    runs = 10
    # variation = "nu"
    variation = "c"
    # variation = "rbf"
    # variation = "lag"

    # MKL Addition
    # variation = "multi"

    # kernel_type = "rbf"
    # kernel_type = "poly"

    # MKL Addition
    kernel_type = "precomputed"

    # dataset = "CAR"
    dataset = "ECO"
    # dataset = "SEED"
    # dataset = "PIMA"
    # dataset = "DERMATOLOGY"
    # dataset = "WINE"
    # dataset = "BIO"
    # dataset = "CONTRA"
    # dataset = "PHO"
    # dataset = "HAY"
    # dataset = "CREDIT"
    # dataset = "WDBC"
    # dataset = "GLASS"
    # dataset = "IONOS"
    # dataset = "HEART"
    # dataset = "SONAR"

    print("Dataset: ", dataset)
    acc_list = []
    for i in np.arange(runs):
        print("Run: ", i)
        acc = svm_var_obj.configure_svms(dataset)
        acc_list.append(acc)
        print("##completed##")
    print(acc_list)
    print("Mean: ",np.mean(acc_list), "\tStd:", (np.std(acc_list)))