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

#%% Load Data
filename = os.path.abspath(
    r"C:\Users\erica\Documents\syde552\Assignment1\assignment1-data\perceptron-data.pkl")
data = pickle.load(open(filename, 'rb'), encoding='latin1')
labels = data['labels'].astype(np.int)
vectors = data['vectors']

#put in dataframe
training_df = pd.DataFrame(columns=["v_1", "v_2", "labels"])
training_df.v_1 = [x for x in vectors[0]]
training_df.v_2 = [x for x in vectors[1]]
training_df.labels = labels

#%% Perceptron

step_fun = lambda x: 0 if x < 0 else 1

training_input = training_df.sample(frac=1)

weights = np.random.randint(2, size=100)

