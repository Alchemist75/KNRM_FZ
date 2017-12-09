import os
import numpy as np 

def process(A,B):
    return A+B,A-B
def data_generator1():

    while True:
        X = np.ones([5000,5000])
        Y = np.zeros([5000,5000])
        X,Y = process(X,Y)
        yield X,Y

def data_generator2():
    
    for X,Y in data_generator1():
        yield (X,Y) 
cnt = 0

while True:
    (X,Y) = next(data_generator1())
    cnt = cnt + 1 
    if (cnt%100000==0):
        print cnt