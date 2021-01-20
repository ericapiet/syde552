# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 23:28:09 2021

SYDE552 Assignment 1 Q1

@author: ericapiet
"""

#%% Import
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from random import random

#%% Load Data
filename = os.path.abspath(
    r"C:\Users\erica\Documents\syde552\Assignment1\assignment1-data\perceptron-data.pkl")
data = pickle.load(open(filename, 'rb'), encoding='latin1')
labels = data['labels'].astype(np.int)
vectors = data['vectors']

#put in dataframe
training_df = pd.DataFrame(columns=["f_1", "f_2", "labels", "outputs"])
training_df.f_1 = [x for x in vectors[0]]
training_df.f_2 = [x for x in vectors[1]]
training_df.labels = labels

#%% Definitions

activation_fun = lambda x: 0 if x < 0 else 1

input_fun = lambda x, w: activation_fun(
    bias + x[0]*weights[0] + x[1]*weights[1])

delta_w = lambda x, y_desired, y_output: gamma*x*(y_desired - y_output)


def change_weight(df):
    weights[0] += delta_w(df.f_1, df.labels, df.outputs)
    weights[1] += delta_w(df.f_2, df.labels, df.outputs)


def perceptron(df, weights, gamma, bias):
    
    #produce predicted output
    df.outputs = training_df.apply(
        lambda x: input_fun(
            np.array([x.f_1, x.f_2]), weights), axis=1)
    
    # adjust the weights
    df.apply(lambda x: change_weight(x), axis=1)
    
    return df, weights
    
    
#%% Training


gamma = 0.01
bias = -1
steps = range(300)
fraction_correct_graph = []
   
# Run 10 times
for i in range(10):

    weights = [random(), random()]
    fraction_correct_steps = [] 
   
    for ii in range(300):
        epoch = training_df.sample(frac=1)
        training_output, new_weights = perceptron(epoch, weights, gamma, bias)
        
        weights = new_weights
        
        # determine fraction of correct outputs
        is_correct = training_output.apply(
            lambda x: 1 if x.labels == x.outputs else 0, axis=1)
        
        fraction_correct_steps.append(is_correct.sum()/100)
            
        fraction_correct_graph.append(fraction_correct_steps)
    
    print(fraction_correct_steps[-1])


#%% Plot

for i in range(10):
    plt.figure(i)
    plt.scatter(steps, fraction_correct_graph[i])



