import math
import random
from tqdm import tqdm
import math
import numpy as np
from submodlib import FacilityLocationFunction, GraphCutFunction, DisparityMinFunction, DisparitySumFunction
import pickle
import time
import os
import timeit
from itertools import permutations
from torchvision import datasets, transforms
import pandas as pd
random.seed(10)
# def generate_sets(num_sets=500, num_data_points=10, num_features=2):
#     sets = []
#     for _ in range(num_sets):
#         set_values = np.random.uniform(0, 1, size=(num_data_points, num_features))
#         sets.append(set_values)
#     return sets

# def facility_location(set_data, subset_fraction_size=0.01):
#     # Create Facility Location Function object
#     obj = FacilityLocationFunction(n=set_data.shape[0], data=set_data, separate_rep=False, mode="dense", metric="cosine")

#     # Maximize the function
#     S = obj.maximize(int(subset_fraction_size * set_data.shape[0]), optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)

#     indices = list(map(lambda tuple: tuple[0], S))
#     print(len(indices))
#     return indices

# num_sets = 500
# num_data_points = 5000
# num_features = 10
# subset_fraction_size = 0.01

# # Generate 500 random sets with 5000 data points and 10 features
# sets = generate_sets(num_sets, num_data_points, num_features)

# # Apply Facility Location Function on each set and get subsets
# subsets = []
# for i, set_data in enumerate(sets):
#     print("set",set_data)
#     indices = facility_location(set_data, subset_fraction_size)

#     #print("subset",subset1)
#     #print("subset",len(subset1[0]))
#     #subsets.append(subset)

# # Print the length of the first subset as an example
# print("Length of the first subset:", len(subsets[0]))


#[1000x10]
#cifar10 and mnist together
#cifar10 dataset

#image -> features -> 
def generate_sets(num_sets=500, num_data_points=10, num_features=2):
    sets = []
    for _ in range(num_sets):
        set_values = np.random.uniform(0, 1, size=(num_data_points, num_features))
        sets.append(set_values)
    return sets

def facility_location(set_data, subset_fraction_size=0.01):
    # Create Facility Location Function object
    #obj = FacilityLocationFunction(n=set_data.shape[0], data=set_data, separate_rep=False, mode="dense", metric="cosine")
    obj = DisparitySumFunction(n=set_data.shape[0], data=set_data, mode="sparse", metric="euclidean",num_neighbors=5)
    #obj = DisparityMinFunction(n=set_data.shape[0], data=set_data, mode="sparse", metric="euclidean",num_neighbors=5)
    #obj = GraphCutFunction(n=set_data.shape[0], data=set_data, separate_rep=False, mode="dense", metric="cosine", lambdaVal =0.1)


    # Maximize the function
    S = obj.maximize(int(set_data.shape[0]-1), optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)

    scores_in_order = {index: 0 for index in range(0,len(set_data))}
    for index, score in S:
        scores_in_order[index] = score
    
    # Extract scores according to the order of indices
    scores_ordered = [scores_in_order[index] for index in range(0,len(set_data))]


    #print(scores_ordered)

    
    return scores_ordered
    #indices = list(map(lambda tuple: tuple[0], S))
    #print(indices)
    # rank_list = [1000] * (len(indices)+1)
    # for i, index in enumerate(indices):
    #     rank_list[index] = i + 1
    #return indices
    


num_sets = 150
num_data_points = 1000
num_features = 10
subset_fraction_size = 0.01

# Generate 500 random sets with 5000 data points and 10 features
sets = generate_sets(num_sets, num_data_points, num_features)

# Apply Facility Location Function on each set and get subsets
subset_indices_dict = {}
for i, set_data in enumerate(sets):
    ranks = facility_location(set_data, subset_fraction_size)
    subset_indices_dict[i] = ranks

# Print the length of the indices of the first subset as an example
#print("Indices of the first subset:", subset_indices_dict[0])

import pickle

# Define filenames for storing sets and ranks
sets_filename = "sets_ds.pkl"
ranks_filename = "scores_ds.pkl"

# Save sets and ranks to disk
with open(sets_filename, 'wb') as f:
    pickle.dump(sets, f)

with open(ranks_filename, 'wb') as f:
    pickle.dump(subset_indices_dict, f)
