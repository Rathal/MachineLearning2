# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:04:45 2019

@author: Daniel Guy (16607811)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import Data
data = pd.DataFrame(data = pd.read_csv("dataset.csv"))

#Provide a summary
for col in data:
    print(data[col].describe())
    print()

print("\n")

#Data Preprocessing
# =============================================================================
#Get size of the data
rows = data['Status'].count()
size = data.size
print("Size of Data: {0} rows; {1} data points".format(rows, size))

#Get number of features
cols = data.columns
print("Number of Features: {0}".format(len(cols)))

#Any Missing Data
print("\nMissing Data: \n{0}".format(data.isnull().sum()))

#Get Data that is Categorical
categorical = data.select_dtypes(exclude=['number'])
print("\nCategorical Data: {0}".format(categorical.columns.values))
# =============================================================================



#Plotting Data
# =============================================================================
sns.boxplot(x="Status", y="Vibration_sensor_1", data=data)
plt.show()

dfStatusNormal = data.loc[data['Status'] == 'Normal']
dfStatusAbnormal = data.loc[data['Status'] == 'Abnormal']
dfStatusNormal['Vibration_sensor_2'].plot.density()
dfStatusAbnormal['Vibration_sensor_2'].plot.density()
plt.legend(labels=['Normal', 'Abnormal'])
plt.xlabel('Vibration Sensor 2')
plt.show()
# =============================================================================


#Normalising Data
# =============================================================================
z = pd.DataFrame()
for col in data:
    if data[col].dtype == object:
        z[col] = data[col]
    else:
        mean = data[col].mean()
        std = data[col].std()
        z[col] = ((data[col] - mean)/std)

#Converting Normal to 1, Abnormal to 0
z = z.replace({'Normal': 1, 'Abnormal': 0})
#print(z.describe())
# =============================================================================

#import sklearn
from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#from sklearn.metrics import r2_score
#from sklearn.metrics import classification_report,confusion_matrix


#Creating the training and test data sets
# =============================================================================
#Defines the Status column as the output for the ML algorithms
outputCol = ['Status']
#Defines the rest of the columns as the predictors
inputCols = list(set(list(z.columns))-set(outputCol))
X = z[inputCols].values
Y = z[outputCol].values

#Splits the data 90:10 train:test.
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1)
# =============================================================================


#ANN
# =============================================================================
def ANN_fit(x, y, epochs):
    mlp = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=epochs)
    mlp.fit(xTrain, yTrain.ravel())
    return mlp
def ANN_predict(mlp, x):
    prediction = mlp.predict(x)
    return prediction


epochs = [1, 25, 50, 100, 250, 500, 1000, 5000] #1, 25, 50, 100, 250, 500
accuracyANN = []
for e in epochs:
    print("Epoch: " + str(e))
    mlp = ANN_fit(xTrain, yTrain, e)
#    predictTest = ANN_predict(mlp, xTest)
    accuracyANN.append(mlp.score(xTest, yTest))

plt.plot(epochs,accuracyANN)
plt.title("ANN Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(labels=['Testing', 'Training '])
plt.show()
# =============================================================================

#Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

trees = [1, 10, 50, 100, 1000, 5000]
leaf = [5, 50]
accuracyTree = []
for l in leaf:
    for tree in trees:
        print("Leaf {0}: Tree {1}".format(l, tree))
        regressor = RandomForestRegressor(n_estimators=tree, min_samples_leaf=l)
        regressor.fit(xTrain, yTrain.ravel())
        accuracyTree.append(regressor.score(xTest, yTest.ravel()))

#print(accuracyTree)
half = len(accuracyTree)//2
accLeaf5 = accuracyTree[:half]
accLeaf50 = accuracyTree[half:]
#print(accLeaf50)

plt.plot(trees,accLeaf5)
plt.plot(trees,accLeaf50)
plt.xscale('log')
plt.xlabel('Trees')
plt.ylabel('Accuracy')
plt.legend(labels=['Leaf = 5', 'Leaf = 50'])
plt.show()
# =============================================================================


#predict_train = mlp.predict(xTrain)
#predict_test = mlp.predict(xTest)

#print(mlp.score(xTrain, yTrain))
#
#print("Train Data:\n")
#print(confusion_matrix(yTrain,predict_train))
#print(classification_report(yTrain,predict_train))
#print("Test data:\n")
#print(confusion_matrix(yTest,predict_test))
#print(classification_report(yTest,predict_test))




# weights =2*np.random.random((2,2)) -1
# 
# for epoch in range(200):
#     layer0 = trainData
#     layer1 = sigmoid(np.dot(layer0,weights[0]))
#     layer2 = sigmoid(np.dot(layer1,weights[1]))
#     
#     layer
# 
# =============================================================================
# =============================================================================
# class NeuralNetwork:
#     def __init__(self, x, y):
#         self.input = x
#         self.weights1 = np.random.rand(self.input.shape[1],500)
#         self.weights2 = np.random.rand(500,2)
#         self.y = y
#         self.output = np.zeros(y.shape)
#     def feedforward(self):
#         self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#         self.output = sigmoid(np.dot(self.layer1, self.weights2))
#     def backpropagation(self):
#         d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_der(self.output)))
#         d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_der(self.output), self.weights2.T) * sigmoid_der(self.layer1)))
#         self.weights1 += d_weights1
#         self.weights2 += d_weights2
# 
# ANN = NeuralNetwork(trainData,trainLabel)
# ANN.feedforward()
# ANN.backpropagation()
# print(ANN.output)
# =============================================================================

# =============================================================================
# #Hyperparameters
# epochs = 100
# layers = 2
# neurons = 250
# lr = 0.05
# 
# def init_param(layers, neurons):
#     param = [None] *layers
#     for i in range(layers):
#         param[i] = np.zeros(neurons)
#     return param
# 
# #Define Sigmoid Functions
# def sigmoid(x):
#     return 1/1(np.exp(-x))
# def sigmoid_der(x):
#     return sigmoid(x)*(1-sigmoid(x))
# 
# 
# weights = init_param(layers, neurons)
# bias = init_param(layers,neurons)
# 
# for epoch in range(epochs):
#     inputs = trainData
#     
#     for l in range(layers):
#         XW = np.dot(trainData, weights) + bias
#     print(XW)
# =============================================================================
    
# =============================================================================