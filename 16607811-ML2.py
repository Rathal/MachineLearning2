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

#sns.distplot(dfStatusNormal['Vibration_sensor_2'])
#sns.distplot(dfStatusAbnormal['Vibration_sensor_2'])
plt.show()
#data['Vibration_sensor_2'].groupby('Status').plot.density()
#plt.show()
#data.boxplot(column='Vibration_sensor_1', by='Status')
#plt.show()
dfStatusNormal['Vibration_sensor_2'].plot.density()
dfStatusAbnormal['Vibration_sensor_2'].plot.density()
plt.legend(labels=['Normal', 'Abnormal'])
plt.show()
# =============================================================================


"""
Normalising Data
z-normalisation
z = (x-mean)/std
"""

# =============================================================================
#Get numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
nData = data.select_dtypes(include=numerics)
z = pd.DataFrame()
for col in data:
    if col == "Status":
        z[col] = data[col]
    else:
        mean = data[col].mean()
        std = data[col].std()
        z[col] = ((data[col] - mean)/std)
print(z.tail())
# =============================================================================




##Normalisation
#Zi = (Xi - Xbar) / Sd


# =============================================================================
# f, (ax_box, ax_hist) = plt.subplots(2, sharex=False, gridspec_kw={"height_ratios": (.5, .5)}) #.15 .85
# sns.boxplot(x="Status", y="Vibration_sensor_1", data=data, ax=ax_box)
# 
# dfStatusNormal = data.loc[data['Status'] == 'Normal']
# dfStatusAbnormal = data.loc[data['Status'] == 'Abnormal']
# 
# sns.distplot(dfStatusNormal['Vibration_sensor_2'], ax=ax_hist)
# sns.distplot(dfStatusAbnormal['Vibration_sensor_2'], ax=ax_hist)
# 
# 
# 
# 
# Attempting to Normalise Data
# 
# #Get numerical only data
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# newData = data.select_dtypes(include=numerics)
# 
# for col in newData:
#     mean = newData[col].mean()
#     std = newData[col].std()
#     z = ((newData[col]-mean)/std)
# =============================================================================
