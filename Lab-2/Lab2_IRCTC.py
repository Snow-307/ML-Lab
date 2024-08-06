# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:26:04 2024

@author: sneha_xqbh6g1
"""

import statistics
import numpy as np
import pandas as pd

def get_data():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price", converters={"Price": int})
    return df
    
def mean_variance(data):
    price = data["Price"]
    popu_mean = statistics.mean(price)
    var = statistics.variance(price)
    print("Variance of price", var)
    print("Mean of price", popu_mean)

    wed_data = data.loc[data["Day"] == "Wed", "Price"]
    wed_mean = statistics.mean(wed_data)
    print("Mean of price for all wednesdays",wed_mean)
    
    apr_data = data.loc[data["Month"] == "Apr", "Price"]
    apr_mean = statistics.mean(apr_data)
    print("Mean of price for April",apr_mean)

def loss_probability(data):
    chg_data = data["Chg%"]
    losses = chg_data.apply(lambda x: x < 0).sum()
    length = len(chg_data)
    loss_prob = losses / length
    print(loss_prob)
    return loss_prob

def prob_profit_on_wed(data):
    wed_data = data.loc[data["Day"] == "Wed", "Chg%"]
    profit_wed = wed_data.apply(lambda x: x > 0).sum()
    length = len(wed_data)
    profit_prob = profit_wed / length
    print("Probability of making profit on wednesday",profit_prob)
    
    #conditional probability
    conditional_prob_wednesday = profit_wed / length
    return conditional_prob_wednesday
    return profit_prob

def scatter_plot(data):
    req_data = data[["Day","Chg%"]]
    req_data.plot(kind="scatter", x="Day", y="Chg%", color='blue', marker='o')

def main():
    data = get_data()
    #mean_variance(data)
    #loss_probability(data)
    #scatter_plot(data)

if __name__ == '__main__':
    main()
