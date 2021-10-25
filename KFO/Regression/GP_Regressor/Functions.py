import numpy as np

import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH


class FunctionHelper:

    def __init__(self, func_type):
        self.true_func_type = func_type

    def get_true_max(self):

        # define y_max for the true functions
        if (self.true_func_type == 'custom'):
            # exp{-(x-2)^2} + exp{-((x-6)^2)/10} + (1/(X^2 +1))
            # true_max = self.get_true_func_value(2.0202)

            # oscillator
            # exp(-x)sin(3x) + 1
            # true_max = self.get_true_func_value(0.15545)

            # complicated oscillator
            # (np.exp(-each_x) * np.sin(1.5 * np.pi * each_x)) + 1
            # true_max = self.get_true_func_value(0.3001)

            # exp(-x)sin(8.pi.x) + 1
            # true_max = self.get_true_func_value(0.061)

            # Gramacy and Lee function sin(10.pi.x/2x)+(x-1)^4; minima = -2.874 @ x=0.144; -sin(10.pi.x/2x)-x-1)^4; maxima = 2.874 @x=0.144
            # in the range [0.5, 2.5] max is 0.869 @x = 0.5486
            # true_max = self.get_true_func_value(0.5486)

            # Standardised y maxima = 1.23880 @ x= 0.024024
            # true_max = self.get_true_func_value(0.024024)

            # Levy function w = 1+(x-1)/4  y = (sin(w*pi))^2 + (w-1)^2(1+(sin(2w*pi))^2) max =0 @x=1.0
            # true_max = self.get_true_func_value(1.0)

            # Benchmark Function exp(-x)*sin(2.pi.x)(maxima = 0.7887), -exp(-x)*sin(2.pi.x) (minima)
            # true_max = self.get_true_func_value(0.22488)

            # Square wave
            # true_max = self.get_true_func_value(0.2)

            #Triangular wave
            # true_max = self.get_true_func_value(0.5)

            # chirpwave
            # true_max = self.get_true_func_value(0.3783)

            # Sinc function
            # true_max = self.get_true_func_value(-10.05)

            # Gaussian Mixture
            # true_max = self.get_true_func_value(2.502)

            # Linear Function
            # true_max = self.get_true_func_value(1.99)

            #Linear Sin Function
            true_max = self.get_true_func_value(9.268)

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

        PH.printme(PH.p1, "True function:",self.true_func_type," \nMaximum is ", true_max)
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

            #oscillator
            # exp(-x)sin(3.pi.x) + 0.3
            # return (np.exp(-x) * np.sin(3 * np.pi * x)) + 1

            # Complicated Oscillator circuit
            # val = 0
            # if x < 10:
            #     val = (np.exp(-x) * np.sin(1.5 * np.pi * x)) + 1
            # elif x > 10 and x <= 20:
            #     x = x - 10
            #     val = (np.exp(-x) * np.sin(1.5 * np.pi * x)) + 1
            # elif x > 20 and x <= 30:
            #     x = x - 20
            #     val = (np.exp(-x) * np.sin(1.5 * np.pi * x)) + 1
            # return val

            # exp(-x)sin(8.pi.x) + 1
            # return (np.exp(-x) * np.sin(8 * np.pi * x)) + 1

            # Benchmark Function exp(-x)*sin(2.pi.x)(maxima), -exp(-x)*sin(2.pi.x) (minima)
            # return (np.exp(-x) * np.sin(2 * np.pi * x))

            # Gramacy and Lee function sin(10.pi.x/2x)+(x-1)^4; minima = -2.874 @ x=0.144; -sin(10.pi.x/2x)-x-1)^4; maxima = 2.874 @x=0.144
            # return -1 * ((((np.sin(10 * np.pi * x))/(2*(x)))) +(x-1) ** 4)

            # Levy function w = 1+(x-1)/4  y = (sin(w*pi))^2 + (w-1)^2(1+(sin(2w*pi))^2) max =0
            # w = -0.5+((x-1)/4)
            # w = 1+((x-1)/4)
            # value = ((np.sin(w * np.pi))**2 + ((w-1)**2)*(1+((np.sin(2*w*np.pi)) ** 2 )))
            # return -1 * value

            # square wave function
            # y = np.array([])
            #
            # for each_x in x:
            #     each_y = np.sin(2 * np.pi * each_x)
            #     if each_y < 0:
            #         each_y = -1
            #     elif each_y > 0:
            #         each_y = 1
            #     else:
            #         each_y = 0
            #     y = np.append(y, each_y)
            # return y.reshape(-1, 1)

            # Triangular wave function
            # return (2 * np.arcsin(np.sin(np.pi * x))) / (np.pi)

            # Chirpwave function
            # y = np.array([])
            # f = 1
            # for each_x in x:
            #     if each_x < 8:
            #         f = 0.35
            #     elif each_x > 8 and each_x <= 15:
            #         f = 1.25
            #     elif each_x > 15 and each_x <= 20:
            #         f = 0.35
            #     val = np.sin(2 * np.pi * f * each_x)
            #     y = np.append(y, val)
            # return y.reshape(-1, 1)

            # Sinc Function
            # return np.sinc(x - 10) + np.sinc(x) + np.sinc(x + 10)

            # Gaussian Mixtures
            # y = np.array([])
            # for each_x in x:
            #     if each_x <= 5:
            #         sig = 0.4
            #         mean = 2.5
            #     elif each_x > 5 and each_x <= 10:
            #         sig = 0.7
            #         mean = 7.5
            #     elif each_x > 10 and each_x <= 15:
            #         sig = 0.6
            #         mean= 12.5
            #     val = 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((each_x - mean) / sig) * (each_x - mean) / sig)
            #     y = np.append(y, val)
            # return y.reshape(-1, 1)

            # Linear function
            # return 0.1 * x + 0.2

            # Linear Sin Function
            return 0.7*x + 1 + np.sin(2*np.pi*x)

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
            # Regression setting
            # value = -1 * value.reshape(-1, 1)
            value = 1 * value.reshape(-1, 1)
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

