#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:11:08 2018

@author: prashantn
"""
#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Load the dataset
#For this example I am using iris dataset available on pandas 

iris_data = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')


#Lets find the unique label values

iris_label_unique = iris_data.Name.unique()

print(iris_label_unique)

#['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
#So here when we create confusion matrix, there will exists multiple
#label fields unlike traditional true /false. Lets explore this confusion 
#matrix

#Seperating features and label
features = iris_data.iloc[:,:-1].values
label = iris_data.iloc[:,[4]].values

#Creating train_test split

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features,
                                                label,
                                            test_size=0.2,
                                            random_state=0)

#Apply LogisticRegressionModel over data


#Create the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

#Check the accuracy score for training & testing data

print("Testing data Accuracy: ",model.score(X_train,y_train))
print("Testing data Accuracy: ",model.score(X_test,y_test))

print(model.predict(np.array([[5.1,3.5,1.4,0.2]])))

#Create confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))

#Method1 - Using Pandas 
y_true_test = pd.Series(y_test.reshape(-1,))
y_pred_test = pd.Series(model.predict(X_test).reshape(-1,))
pd.crosstab(y_true_test, 
            y_pred_test, 
            rownames=['True'], 
            colnames=['Predicted'], 
            margins=True)

#Method2 - Visualizing using seaborn (Static)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))
import seaborn as sns

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Iris-setosa', 'Iris-versicolor','Iris-virginica']); 
ax.yaxis.set_ticklabels(['Iris-setosa', 'Iris-versicolor','Iris-virginica']);


#Method2 - Visualizing using seaborn (Static)
import seaborn as sns

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['']+iris_label_unique); 
ax.yaxis.set_ticklabels(['']+iris_label_unique);



