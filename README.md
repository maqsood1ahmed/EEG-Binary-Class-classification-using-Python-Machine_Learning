
# Devloped By Muhammad Salman, Fahad Shaikh and Maqsood Ahmed
#
#
# EEG-Binary_Class_classification_using_Python_Machine_Learning

# Note:-
      Following things should must be installed to run this project
         ==>Python 2.7 
         ==>kivy
         ==>Pandas library to handle data frame
         ==>Numpy library to handle array and other scientific calculations
         ==>sklearn for machine learning libraries here we are using SVM rbf functions
         ==>Emotiv SDK premium Edition 32bit version libraries 

# For Data collection and live prediction following libraries are required which comes with Emotiv Epoc SDK Premium
     # edk.dll
     # edk_utils.dll
     # edk.lib
     # edk_utils.lib
    
# To Run this project
      step#1 :- Open Fourth_Main_GUI.py
      step#2 :- Create or Load profile
      step#3:-  Click on Train Button and start training and follow the instructions
      step#4 :- After more than 50 trials you should click on Live Button which will create a model
      step#5 :- Click on Start Button for live prediction of data but first provide 20 seconds neutral though data

Accuracy Achieved from this model is about 60% which can be increased by applying Neural Network and Deep Learning Models
Students or Developers can use this repository as a reference for their project they just need to change grid_searchcv() method in Second_ProcessData.py file to apply different models based on their requirement.

For Live_Prediction 5second data is taken to predict specific thought
