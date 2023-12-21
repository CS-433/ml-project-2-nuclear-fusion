[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)

# Machine Learning Project 2 :  'Characterization and automatic differentiation between minor and major disruptions'

This is part of the Project 2 of the CS433-Machine Learning course given at the EPFL Fall 2023.

## Team NUCLEAR-FUSION : 
- Kaan Uçar
- Elias Nicolas Naha
- Riccardo Carpineto

## Project Description 

The aim of the project is to predict minor and major disruptions based on personal features. To perform this classification task we need to perfom the following :
- Preprocessing physical data from the Tokamak sensors
- Implement Deep Learning and Advanced machine learning models
- Hyperparameter tuning and joint-model pipelining

We then select the most performing one. The performance is measured by the F1 score which is a good perfomance value for imbalanced data.
In our case the data is unbalanced over the plasma events : 899 time-series windows of 20 datapoints; 642 for no-event label,  207 for minor events and 50 for major events.
The best run we achieved a weighted 0.9684 F1 score using ResNet Architecture Classifier to classify theses windows.

## Structure of Repository :

- `figure` folder : contains plot from grid searches for hyperparameter tuning from RF and SVM, as well as the learning curve from SVM 

- `data` folder : contains all the data needed to run the code, the raw data from the experiments as well as the windows centered on the events given

- `models` : contains models and saved weights from the ResNet and NN models to avoid to train the model again

- `code` folder: contains all the executable files to run the code 

    - `GMM.ipynb` : executable file that reproduces our implementation of a Gaussian Mixture Model (GMM), an unsupervised probabilistic model 

    - `NN_general.ipynb` : executable file implementing the neural network for the basic Neural Network for classification

    - `NN_window.ipynb` : executable file that helps forming the windows thanks to the use of a tuned Neural Network, with label 0s and 1s

    - `RNN.ipynb` : executable file implementing a Recurrent Neural Network used to classify the different labels window by window 
				
    - `ResNet.ipynb` : executable file implementing a Residual Neural Network used to classify the different labels window by window, leading the best results from the others models
	
    - `SVM.ipynb` : executable file implementing SVM
    
    - `RF.ipynb` : executable file implementing Random Forest
      
    - `scaler.pkl` : normalizer used in the ResNet to pre-process the data 
    
    - `window_creation.ipynb` : executable file creating the windowed data from time series data. This data is saved in the data folder.

    - `pipeline.ipynb` : file containing the pipeline showing the process for predicting labels for a raw experiment, implementing in a first part the 'NN_window' to form the differents windows, and then apply the ResNet to predict the labels from theses.

    - `test.ipynb` : folder containing all the data


## Instructions to run :

To try out a model you can run any `.ipynb` model file with the presence of the `data` folder. The models used for the joint-model are 'NN_window.ipynb' to form windows and then `ResNet.ipynb` to predict their labels. You can run them separately to observe their accuracy, and then run `pipeline.ipynb` to see how the pipeline works in details.
