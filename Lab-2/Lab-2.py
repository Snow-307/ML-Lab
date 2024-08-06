# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:01:49 2024

@author: sneha_xqbh6g1
"""

import numpy as np
import pandas as pd

def get_data():

    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data", usecols=["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"])
    A = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]] 
    C = df[["Payment (Rs)"]] 
    return A,C

def dim(df):
    print("Dimensionality:", df.shape)
    array = df.to_numpy()
    print("Rank:", np.linalg.matrix_rank(array))

def pseudo_inverse(m):
    inv = np.linalg.pinv(m)
    print("Pseudoinverse:\n", inv)
    return(inv)

def calculate_cost(I,C):
    print(np.matmul(I,C))
    
def rich_poor(Pay,A):
    array = Pay.to_numpy()
    rp = np.where(array>200,"Rich","Poor")
    for payment, status in zip(array, rp):
        print(f"{payment}{status}")
    
    
    

def main():
    A,C= get_data()
    dim(A)
    inverse = pseudo_inverse(A)
    calculate_cost(inverse,C)
    rich_poor(C)
    

if __name__ == '__main__':
    main()
