# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:50:56 2024

@author: sneha_xqbh6g1
"""

import statistics
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats

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

    print('Mean:\n', mean)
    print('Variance:\n', variance)
    print('Standard Deviation:\n', std_dev)    
    return mean


def fill_na(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)
    
    
def main():
    data = get_data()
    categorical(data)
    outliers(data)
    mean= calc_stats()
    fill_na(data,mean)
    

if __name__ == '__main__':
    main()

