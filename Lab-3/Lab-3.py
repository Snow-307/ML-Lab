# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:20:21 2024

@author: sneha_xqbh6g1
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Amrita\Sem 5\ML\hm\image_data3.csv')

class_ajanta = data[data['label'] == 'Ajanta Caves']
class_mysore = data[data['label'] == 'mysore_palace']
class_gateway = data[data['label'] == 'Gateway of India']


features_ajanta = class_ajanta.drop('label', axis=1)
features_mysore = class_mysore.drop('label', axis=1)
features_gateway = class_gateway.drop('label', axis=1)


features_ajanta = features_ajanta.to_numpy()
features_mysore = features_mysore.to_numpy()
features_gateway = features_gateway.to_numpy()


def centroids():
    centroid_ajanta = np.mean(features_ajanta, axis=0)
    centroid_mysore = np.mean(features_mysore, axis=0)
    centroid_gateway = np.mean(features_gateway, axis=0)
    print("Centroid of Ajanta Caves:", centroid_ajanta)
    print("Centroid of Mysore Palace:", centroid_mysore)
    print("Centroid of Gateway of India:", centroid_gateway)

    dist_ajanta_mysore = np.linalg.norm(centroid_ajanta - centroid_mysore)
    dist_ajanta_gateway = np.linalg.norm(centroid_ajanta - centroid_gateway)
    dist_mysore_gateway = np.linalg.norm(centroid_mysore - centroid_gateway)
    
    print("Euclidean Distance (Ajanta Caves & Mysore Palace):", dist_ajanta_mysore)
    print("Euclidean Distance (Ajanta Caves & Gateway of India):", dist_ajanta_gateway)
    print("Euclidean Distance (Mysore Palace & Gateway of India):", dist_mysore_gateway)


def spread():
    spread_ajanta = np.std(features_ajanta, axis=0)
    spread_mysore = np.std(features_mysore, axis=0)
    spread_gateway = np.std(features_gateway, axis=0)
    print("Spread of Ajanta Caves:", spread_ajanta)
    print("Spread of Mysore Palace:", spread_mysore)
    print("Spread of Gateway of India:", spread_gateway)

def train_test():
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def knn(X_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    return neigh


def evaluate_performance(neigh, X_test, y_test):
    accuracy = neigh.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    y_pred = neigh.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")


centroids()
spread()

X_train, X_test, y_train, y_test = train_test()
neigh = knn(X_train, y_train)
evaluate_performance(neigh, X_test, y_test)



