# Machine_learning_project
For runnig the code you need to run main.py file. This file uses one of the class written by me. This class is located in \ML_FAB_Classes\NN_Prep.py. 
The class does neural network training. 
The main.py is made of 5 parts which each part is elaborated hereafter

Part 1: Importing libraries
It imports the required libraries as well as the class that is written by me for neural network training

Part 2:load the data object
This section loads the data from an xdf file and coverts it to a panda data frame format

Part 3:data initialization
This section removes noises from the data and also does hot encoder operation on the discreet features 

Part 4: Normalization for ML
This section normalizes both the features and continuous labels  

Part 5: Training
This section trains the neural networks and get the trained model  

Part 6: Results and plots
This section shows how much error the model has based on all available data ( which includes the train and test data)
It also plots the labels of all available data and compare it with predicted labels come from the trained model
