# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:04:12 2024

@author: sneha_xqbh6g1
"""
import numpy as np

#Q1
def count_pair_10():   
    l = [2,7,4,1,3,6]
    length = len(l)
    count = 0
    for i in range (0,length):
        for j in range (0,length):
            if l[i]+l[j]==10:
                count=count+1
            j=j+1 
        i=i+1             
    print ("Pairs with sum equal to 10:", count)
      
#Q2

def range(l):
    length1 = len(l)
    if length1<3:
        print("Error: Range determination not possible")
    else:
        min=l[0]
        i=0
        while i<length1:
            if l[i]<min:
                min=l[i]
            i=i+1
        
        max=l[0]
        i=0
        while i<length1:
            if l[i]>max:
                max=l[i]
            i=i+1
            
        range_of_list=max-min
        print(range_of_list)
        
#Q3
def matrix_exponent(matrix,num):
    x,y=matrix.shape
    result=matrix
    exp_count=1
    if(x==y):
        while exp_count<=num:
            result = np.dot(result,matrix)
            exp_count=exp_count+1
        print(result)
        
        
#Q4
def most_occuring_char(s):
    length = len(s)
    i=0
    count_max=0
    letter="a"
    while(i<length):
        count_check = s.count(s[i])
        if(count_check>count_max):
            count_max=count_check
            letter = s[i]
        i=i+1
    print("Most occuring character:",letter, "\nOccurence count:", count_max)
       
       


#main
def main():
    '''
    count_pair_10()
    list = [5,3,8,1,0,4]
    range(list)
    
    m=np.array([[1,2,3],[3,6,4],[7,2,5]])
    matrix_exponent(m,3)
    '''
    string="hippopotamus"
    most_occuring_char(string)


if __name__ == '__main__': 
    main() 
