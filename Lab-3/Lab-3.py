# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:46:29 2024

@author: sneha_xqbh6g1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score




def class_mean(class1,class2):
    
    centroid1 = np.mean(class1, axis=0)
    centroid2 = np.mean(class2, axis=0)
    Euclidean_dist = np.linalg.norm(centroid1 - centroid2)
    
    print("Centroid of class 1:\n",centroid1)
    print("\n Centroid of class 2:\n",centroid2)
    print("\n Euclidean Disctance:",Euclidean_dist)

def spread(class1,class2):
    spread1 = np.std(class1, axis=0)
    spread2 = np.std(class2, axis=0)
    print("Spread of Class 1:\n",spread1)
    print("Spread of Class 2:\n",spread2)

def histo():
    feature_data = data['BMI']
    plt.hist(feature_data, bins=10)  # Adjust bins as needed
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.show()
    mean_bmi = np.mean(feature_data)
    variance_bmi = np.var(feature_data)
    
    print("Mean of BMI:",mean_bmi)
    print("Variance of BMI:",variance_bmi)


def minkowski_q():
    required = ['Age', 'BMI', 'Sleep Duration (Hours)']
    vector1 = data.loc[0, required].values
    vector2 = data.loc[1, required].values
    
    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)
    
    distances = [minkowski(vector1, vector2, p=r) for r in range(1, 11)]
    
    plt.plot(range(1, 11), distances)
    plt.xlabel('r')
    plt.ylabel('Minkowski Distance')
    plt.show()

def train_test():
    
    X = data[['Age', 'Gender', 'Race', 'BMI', 'Smoking', 'Heavy Drinking', 'Sleep Duration (Hours)', 'Arthritis', 'Liver Condition', 'Parental Osteoporosis']]
    y = data['Osteoporosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return(X_train, X_test, y_train, y_test)

def knn(X_train,y_train):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    return neigh
    
def check_accu(neigh):
    accuracy = neigh.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    predictions = neigh.predict(X_test)
    print(f"Predictions: {predictions}")
    specific_prediction = neigh.predict([X_test.iloc[0]])
    print(f"Prediction for first test vector: {specific_prediction}")
    accuracies = []
    k_values = range(1, 12)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracies.append(knn.score(X_test, y_test))
        
        plt.plot(k_values, accuracies, marker='o')
        plt.title('Accuracy vs. k in kNN')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.show()
        
    y_pred = neigh.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    recall = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    
    
    
    
data = pd.read_csv('data.csv')
class1_data = data[data['Osteoporosis'] == "Yes"]  
class2_data = data[data['Osteoporosis'] == "No"]  


features_class1 = class1_data[['Age', 'Gender', 'Race', 'BMI', 'Smoking', 'Heavy Drinking', 'Sleep Duration (Hours)', 'Arthritis', 'Liver Condition', 'Parental Osteoporosis']]
features_class2 = class2_data[['Age', 'Gender', 'Race', 'BMI', 'Smoking', 'Heavy Drinking', 'Sleep Duration (Hours)', 'Arthritis', 'Liver Condition', 'Parental Osteoporosis']]

numeric_class1 = class1_data.select_dtypes(include=['number'])
numeric_class2 = class2_data.select_dtypes(include=['number'])

'''
class_mean(numeric_class1, numeric_class2)
spread(numeric_class1, numeric_class2)
histo()
minkowski_q()
'''

X_train, X_test, y_train, y_test = train_test()
neigh=knn(X_train,y_train)
check_accu(neigh)


