# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:50:56 2024

@author: sneha_xqbh6g1
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statistics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

def get_data():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return df

def categorical(data):
    #categorical_cols = data[["sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant","thyroid surgery","I131 treatment","query hypothyroid","query hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH measured","T3 measured","TT4 measured","T4U measured","FTI measured","TBG measured","referral source"]]
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        data[col] = data[col].astype(str)
      
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    data = pd.get_dummies(data, columns=categorical_cols)
    return data


def dataranges(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    print(data[numeric_cols].agg(['min', 'max']))


def outliers(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    z_scores = stats.zscore(data[numeric_cols])
    outliers = (abs(z_scores) > 3).sum(axis=0)
    print(f'Number of outliers in each numeric column:\n{outliers}')


def calc_stats(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    mean = data[numeric_cols].mean()
    variance = data[numeric_cols].var()
    std_dev = data[numeric_cols].std()

    print('Mean:', mean)
    print('\n Variance:', variance)
    print('\n Standard Deviation:', std_dev)    
    return mean


def fill_na(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

def normalize_data(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data


def calculate_jaccard_and_smc_V(vector1, vector2):
    v1 = pd.Series(vector1)
    v2 = pd.Series(vector2)

    f11 = ((v1 == 1) & (v2 == 1)).sum()
    f01 = ((v1 == 0) & (v2 == 1)).sum()
    f10 = ((v1 == 1) & (v2 == 0)).sum()
    f00 = ((v1 == 0) & (v2 == 0)).sum()
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0    
    smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0
    print(f'Jaccard Coefficient: {jc}')
    print(f'Simple Matching Coefficient: {smc}')
    return jc, smc


def calculate_cosine_similarity_V(vector1, vector2):

    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    cosine_sim = cosine_similarity(vector1, vector2)[0][0]
    print(f'Cosine Similarity: {cosine_sim}')
    return cosine_sim   

def calculate_jaccard_and_smc(matrix):
    num_vectors = matrix.shape[0]
    jc_matrix = np.zeros((num_vectors, num_vectors))
    smc_matrix = np.zeros((num_vectors, num_vectors))
    
    for i in range(num_vectors):
        for j in range(i, num_vectors):
            vector1 = matrix[i]
            vector2 = matrix[j]
            
            f11 = np.sum((vector1 == 1) & (vector2 == 1))
            f01 = np.sum((vector1 == 0) & (vector2 == 1))
            f10 = np.sum((vector1 == 1) & (vector2 == 0))
            f00 = np.sum((vector1 == 0) & (vector2 == 0))
            
            # Calculate Jaccard Coefficient
            jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
            jc_matrix[i, j] = jc
            jc_matrix[j, i] = jc  # Symmetric matrix
            
            # Calculate Simple Matching Coefficient
            smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0
            smc_matrix[i, j] = smc
            smc_matrix[j, i] = smc  # Symmetric matrix
    
    return jc_matrix, smc_matrix

def calculate_cosine_similarity_matrix(matrix):
    cosine_sim_matrix = cosine_similarity(matrix)
    return cosine_sim_matrix

def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


 
def main():
    data = get_data()
    categorical(data)
    outliers(data) 
    mean= calc_stats(data)
    fill_na(data)
    subset_data = data.head(20)
    matrix = subset_data.values
    binary_cols = data.columns[data.nunique() == 2]
    binary_data = data[binary_cols]
    
    vector1 = binary_data.iloc[0]
    vector2 = binary_data.iloc[1]
    
    jc, smc = calculate_jaccard_and_smc_V(vector1, vector2)
    cosine_sim = calculate_cosine_similarity_V(vector1, vector2)
    
    jc_matrix, smc_matrix = calculate_jaccard_and_smc(matrix)
    cosine_sim_matrix = calculate_cosine_similarity_matrix(matrix)

    plot_heatmap(jc_matrix, 'Jaccard Coefficient Heatmap', 'jaccard_heatmap.png')
    plot_heatmap(smc_matrix, 'Simple Matching Coefficient Heatmap', 'smc_heatmap.png')
    plot_heatmap(cosine_sim_matrix, 'Cosine Similarity Heatmap', 'cosine_heatmap.png')

if __name__ == '__main__':
    main()

