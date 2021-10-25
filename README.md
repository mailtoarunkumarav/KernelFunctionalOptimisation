# KernelFunctionalOptimisation
Code base for NeurIPS 2021 publication titled Kernel Functional Optimisation (KFO)

We have conducted all our experiments in a server with the following specifications.
  • RAM: 16 GB
  • Processor: Intel (R) Xeon(R) W-2133 CPU @ 3.60GHz
  • GPU: NVIDIA Quadro P 1000 4GB + 8GB Memory
  • Operating System: Ubuntu 18.04 LTS Bionic

We recommend installing the following dependencies for the smooth execution of the modules.
  • Python - 3.6
  • scipy - 1.2.1
  • numpy - 1.16.2
  • pandas - 0.24.2
  • scikit-learn - 0.21.2

For running the experiments, please follow the steps mentioned below.

  1. Navigate to the directory named “KFO”, containing the source files required for conducting the experiments.
    
  2. For GP regression experiments:
    (a) Navigate to “Regression” folder
        $ cd Regression
        
    (b) Specify the experimental parameters
        $ python KerOptWrapper.py -d <dataset>
        
        <dataset> is the dataset to be used in the regression (auto, fertility, concreteslump, yacht, · · · )
        For example: $ python KerOptWrapper.py -d auto
          
  3. For SVM classification experiments:
   
    (a) Navigate to “Classification” folder
        $ cd Classification
        
    (b) Specify the experimental parameters
        $ python BayesianOptimisationWrapper.py -d <dataset>
        <dataset> is the dataset to be used in the classification (seeds, wine, heart, ionos, sonar,· · · )
        For example: $ python BayesianOptimizationWrapper.py -d seeds
          
  4. For Synthetic experiments:
   
    (a) Navigate to “Regression/Synthetic” folder
        $ cd Regression/Synthetic
        
    (b) Specify the experimental parameters
        $ python KerOptWrapper.py -f <function_name>
        For example: $ python KerOptWrapper.py -f Triangular
          

Note: The current version of the SVM implementation used in the classification experiments might consume more time
in training, depending on the dataset and the computational power of the server being used.
