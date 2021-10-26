import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
import datetime
sys.path.insert(0, 'HKFKO')
from KerOptWrapper import KernelOptimizationWrapper

import sys
sys.path.append("..")
from HelperUtility.PrintHelper import PrintHelper as PH


class FunctionHelperBO:



    def __init__(self, func_type, cmd_inputs):
        self.true_func_type = func_type
        self.cmd_inputs = cmd_inputs
        if(func_type =="svm"):
            # #Iris data

            # type = 'iris'
            # type = 'wine'
            type = 'wdbc'
            # type = 'vehicle'

            if(type == 'iris'):
                PH.printme(PH.p3, "Working with IRIS Data")
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
                # Assign colum names to the dataset
                colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
                # Read dataset to pandas dataframe
                irisdata = pd.read_csv(url, names=colnames)
                D = irisdata.drop('Class', axis=1)
                f = irisdata['Class']

            elif(type == 'wdbc'):
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
                # bcdata = pd.read_csv("Dataset/wdbc.data")
                PH.printme(PH.p3, "Working with WDBC Data")
                bcdata = pd.read_csv(url)
                D = bcdata.drop(bcdata.columns[[0, 1]], axis=1)
                f = bcdata.iloc[:, 1]

            elif (type == 'wine'):
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
                PH.printme(PH.p3, "Working with wine Data")
                winedata = pd.read_csv(url)
                D = winedata.drop(winedata.columns[[0]], axis=1)
                f = winedata.iloc[:, 0]

            elif (type == 'vehicle'):
                files = os.listdir("Dataset/Vehicle")
                PH.printme(PH.p3, "working with vehicle data")
                total_data_frame = pd.DataFrame()
                for each_file in files:
                    each_data_frame = pd.read_csv("Dataset/Vehicle/" + str(each_file), header=None, sep=r'[ ]', engine='python')
                    total_data_frame = pd.concat([total_data_frame, each_data_frame], ignore_index=True)

                D = total_data_frame.drop(total_data_frame.columns[[-1]], axis=1)
                f = total_data_frame.iloc[:, -1]


            self.D_train, self.D_test, self.f_train, self.f_test = train_test_split(D, f, test_size=0.20)


    def get_true_max(self):

        # define y_max for the true functions
        if (self.true_func_type == 'custom'):
            # exp{-(x-2)^2} + exp{-((x-6)^2)/10} + (1/(X^2 +1))
            # true_max = self.get_true_func_value(2.0202)

            # exp(-x)sin(3x) + 0.3
            # true_max = self.get_true_func_value(0.15545)

            # exp(-x)sin(8.pi.x) + 1
            # true_max = self.get_true_func_value(0.061)

            # Gramacy and Lee function sin(10.pi.x/2x)+(x-1)^4; minima = -2.874 @ x=0.144; -sin(10.pi.x/2x)-x-1)^4; maxima = 2.874 @x=0.144
            # in the range [0.5, 2.5] max is 0.869 @x = 0.5486
            true_max = self.get_true_func_value(0.5486)

            # Standardised y maxima = 1.23880 @ x= 0.024024
            # true_max = self.get_true_func_value(0.024024)

            # Levy function w = 1+(x-1)/4  y = (sin(w*pi))^2 + (w-1)^2(1+(sin(2w*pi))^2) max =0 @x=1.0
            # true_max = self.get_true_func_value(1.0)

            # Benchmark Function exp(-x)*sin(2.pi.x)(maxima = 0.7887), -exp(-x)*sin(2.pi.x) (minima)
            # true_max = self.get_true_func_value(0.22488)

        elif (self.true_func_type == 'sin'):
            true_max = self.get_true_func_value(1.57079)

        elif (self.true_func_type == 'branin2d'):
            # self.y_true_max = 0.397887
            true_max = self.get_true_func_value(np.matrix([[9.42478, 2.475]]))

        elif (self.true_func_type == 'sphere'):
            # self.y_true_max = 0
            true_max = self.get_true_func_value(np.matrix([[0, 0]]))

        elif (self.true_func_type == 'hartmann3d'):
            # self.y_true_max = 3.86278
            # x = [0.114614, 0.555649, 0.852547]
            true_max = self.get_true_func_value(np.matrix([[0.114614, 0.555649, 0.852547]]))

        elif (self.true_func_type == 'hartmann6d'):
            # self.y_true_max = 3.32237
            # x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
            true_max = self.get_true_func_value(np.matrix([[0.20169, 0.150011, 0.476874, 0.275332,
                                                                   0.311652, 0.6573]]))

        elif(self.true_func_type == 'syn2d'):
            true_max = self.get_true_func_value(np.matrix([0.224174, 0.223211]))

        elif (self.true_func_type == 'levy2d'):
            true_max = self.get_true_func_value(np.matrix([1,1]))

        elif (self.true_func_type == 'ackley2d'):
            true_max = self.get_true_func_value(np.matrix([0,0]))

        elif (self.true_func_type == 'egg2d'):
            true_max = self.get_true_func_value(np.matrix([512,404.25425425]))

        elif (self.true_func_type == 'michalewicz2d'):
            true_max = self.get_true_func_value(np.matrix([2.20446091, 1.56922396]))

        elif (self.true_func_type == 'svm'):
            true_max = 1

        elif (self.true_func_type == 'HKFKO'):
            true_max = 1

        PH.printme(PH.p3, "True function:",self.true_func_type," \nMaximum is ", true_max)
        return true_max

    # function to evaluate the true function depending on the selection
    def get_true_func_value(self, x):
        if (self.true_func_type == 'sin'):
            return np.sin(x)

        elif (self.true_func_type == 'cos'):
            return np.cos(x)

        elif (self.true_func_type == 'custom'):
            # exp{-(x-2)^2} + exp{-((x-6)^2)/10} + (1/(X^2 +1))
            # return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)

            # exp(-x)sin(3.pi.x) + 0.3
            # return (np.exp(-x) * np.sin(3 * np.pi * x)) + 0.3

            # exp(-x)sin(8.pi.x) + 1
            # return (np.exp(-x) * np.sin(8 * np.pi * x)) + 1

            # Benchmark Function exp(-x)*sin(2.pi.x)(maxima), -exp(-x)*sin(2.pi.x) (minima)
            # return (np.exp(-x) * np.sin(2 * np.pi * x))

            # Gramacy and Lee function sin(10.pi.x/2x)+(x-1)^4; minima = -2.874 @ x=0.144; -sin(10.pi.x/2x)-x-1)^4; maxima = 2.874 @x=0.144
            return -1 * ((((np.sin(10 * np.pi * x))/(2*(x)))) +(x-1) ** 4)

            # Levy function w = 1+(x-1)/4  y = (sin(w*pi))^2 + (w-1)^2(1+(sin(2w*pi))^2) max =0
            # w = -0.5+((x-1)/4)
            # w = 1+((x-1)/4)
            # value = ((np.sin(w * np.pi))**2 + ((w-1)**2)*(1+((np.sin(2*w*np.pi)) ** 2 )))
            # return -1 * value

        elif (self.true_func_type == 'branin2d'):
            # branin 2d fucntion
            # a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
            # y = a * (x2 - b * x1 **2 + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s
            x1 = x[:, 0]
            x2 = x[:, 1]
            a = 1;
            b = 5.1 / (4 * (np.pi ** 2));
            c = 5 / np.pi;
            r = 6;
            s = 10;
            t = 1 / (8 * np.pi)
            value = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
            value = -1 * value.reshape((-1, 1))
            return value

        elif (self.true_func_type == 'sphere'):
            # simple sphere equation
            # Z = X**2 + Y**2
            x1 = x[:, 0]
            x2 = x[:, 1]
            value = (x1 ** 2 + x2 ** 2)
            value = -1 * value.reshape(-1, 1)
            return value

        elif (self.true_func_type == 'hartmann3d'):

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A_array = [[3, 10, 30],
                       [0.1, 10, 35],
                       [3, 10, 30],
                       [0.1, 10, 35]
                       ]
            A = np.matrix(A_array)

            P_array = [[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]
                       ]

            P = np.matrix(P_array)
            P = P * 1e-4

            sum = 0
            for i in np.arange(0, 4):
                alpha_value = alpha[i]
                inner_sum = 0
                for j in np.arange(0, 3):
                    inner_sum += A.item(i, j) * ((x[:, j] - P.item(i, j)) ** 2)
                sum += alpha_value * np.exp(-1 * inner_sum)
            # extra -1 is because we are finding maxima instead of the minima f(-x)
            value = (-1 * -1 * sum).reshape(-1, 1)
            return value

        elif (self.true_func_type == 'hartmann6d'):

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A_array = [[10, 3, 17, 3.50, 1.7, 8],
                       [0.05, 10, 17, 0.10, 8, 14],
                       [3, 3.5, 1.7, 10, 17, 8],
                       [17, 8, 0.05, 10, 0.1, 14]
                       ]
            A = np.matrix(A_array)

            P_array = [[1312, 1696, 5569, 124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091, 381]
                       ]
            P = np.matrix(P_array)
            P = P * 1e-4

            sum = 0
            for i in np.arange(0, 4):
                alpha_value = alpha[i]
                inner_sum = 0
                for j in np.arange(0, 6):
                    inner_sum += A.item(i, j) * ((x[:, j] - P.item(i, j)) ** 2)
                sum += alpha_value * np.exp(-1 * inner_sum)
            # extra -1 is because we are finding maxima instead of the minima f(-x)
            value = (-1 * -1 * sum).reshape(-1, 1)
            return value

        elif(self.true_func_type == "syn2d"):
            ## Custom synthetic function exp(-x)*sin(2*pi*x)converted from 1d to 2d max =0.6219832327103764 @x= 0.224174, 0.223211
            x1 = x[:, 0]
            x2 = x[:, 1]
            value = (np.exp(-x1) * np.sin(2 * np.pi * x1)) * (np.exp(-x2) * np.sin(2 * np.pi * x2))
            value = value.reshape((-1, 1))
            return value

        elif (self.true_func_type == "levy2d"):
            ## Levy 2d function minima is 0 at X = (1,1)

            X1 = x[:, 0]
            X2 = x[:, 1]

            w1 = 1 + ((X1 - 1) / 4)
            w2 = 1 + ((X2 - 1) / 4)

            value = ((np.sin(np.pi * w1)) ** 2) + ((w1 - 1) ** 2) * (1 + 10 * ((np.sin((np.pi * w1) + 1)) ** 2)) + ((w2 - 1) ** 2) * (
                        1 + ((np.sin(2 * np.pi * w2)) ** 2))

            value = (-1* value).reshape((-1, 1))
            return value

        elif (self.true_func_type == "ackley2d"):
            ## Ackley 2d function minima is 0 at X = (0,0)

            X1 = x[:, 0]
            X2 = x[:, 1]

            a = 20
            b = 0.2
            c = 2 * np.pi
            value = (-20 * np.exp(-0.2 * np.sqrt(0.5 * (X1 ** 2 + X2 ** 2))) - np.exp(
                0.5 * (np.cos(2 * np.pi * X1) + np.cos(2 * np.pi * X2))) + 20 + np.exp(1))

            value = (-1* value).reshape((-1, 1))
            return value

        elif (self.true_func_type == "egg2d"):
            ## egg holder 2d function  Maxima = 959.64008971 @ x =[512,404.25425425]
            X1 = x[:, 0]
            X2 = x[:, 1]

            value =  (-(X2 + 47) * np.sin(np.sqrt(np.abs(X2 + (X1/2) + 47))) - X1 * np.sin(np.sqrt(np.abs(X1 - (X2+47)))))

            value = (-1* value).reshape((-1, 1))
            return value

        elif (self.true_func_type == "michalewicz2d"):
            # Michalewicz maxima = 1.80116404 @ x = [2.20446091 1.56922396]
            X1 = x[:, 0]
            X2 = x[:, 1]

            value =  np.sin(X1) * ((np.sin(((X1 ** 2) / np.pi))) ** 20) + np.sin(X2) * ((np.sin((2 * (X2 ** 2) / np.pi))) ** 20)

            value = (1* value).reshape((-1, 1))
            return value

        elif(self.true_func_type == "svm"):

            c = np.power(10,x[0])
            g = np.power(10,x[1])
            # c = x[0]
            # g = x[1]

            # PH.printme(PH.p3, "Original", x[0],x[1],"\tc,g being queried is ", c, g )
            svclassifier = SVC(kernel='rbf', C=c, gamma=g, random_state=42)
            svclassifier.fit(self.D_train, self.f_train)
            accuracy = svclassifier.score(self.D_test, self.f_test)
            return accuracy


        elif(self.true_func_type == "HKFKO"):

            PH.printme(PH.p3, "\n\n***********#### Running HKFKO for:", x, "##################***********\n\n")
            timenow = datetime.datetime.now()
            stamp = timenow.strftime("%H%M%S_%d%m%Y")

            ker_opt_wrapper_obj = KernelOptimizationWrapper()
            mean_accuracy = ker_opt_wrapper_obj.kernel_wrapper(stamp, x, self.cmd_inputs)
            return mean_accuracy
