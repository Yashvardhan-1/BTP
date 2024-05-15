# %%
import math
import numpy as np
from submodlib import FacilityLocationFunction, GraphCutFunction, DisparityMinFunction, DisparitySumFunction
import pickle
from itertools import permutations, combinations
from torchvision import datasets
import pandas as pd
import os

# %%
dataset = "cifar10"

train_dataset = datasets.CIFAR10(root="../../data", train=True, download=True)
test_dataset = datasets.CIFAR10(root="../../data", train=False, download=True)


# %%
# import sys
# sys.path.append('..')
# from utils.feature_extrater.cifar_feature_extractor import extract_features_threaded_worker, extract_features_resnet_threaded_cifar

# %%
# def generate_permutations(data):
#     """Generates all permutations of a list."""
#     for permutation in permutations(data):
#         yield permutation

# def get_all_ordered_pairs(original_list):
#     """
#     This function takes a list and returns all ordered pairs (lists of 2 elements)
#     formed from the original list.
#     """
#     for per in permutations(original_list, 2):
#         yield per

# %%
metric = "cosine"

# %%
which_exp = 4

# %%
# import argparse
# parser = argparse.ArgumentParser(description="Program to generate sequential subset of data",
#                                  formatter_class=argparse.RawTextHelpFormatter)

# parser.add_argument("--exp", type=int,
#                     default=4,
#                     help="""
#                         Why you want to generate subset?
#                         2: parallel_seq_data_gen
#                         4: seq_data_gen
#                         5: maa chudaye duniya wale
#                     """,
#                     required=True)

# # %%
# args = parser.parse_args()
# which_exp = args.exp

# %%

# which_exp = input("Why you want to generate subset?\
#                    2: paraller_seq_data_geb\
#                    4: seq_data_gen\
#                    5: maa chudaye duniya wale")
# which_exp = int(which_exp)

# %%
func_list = ["facility-location", "disparity-min",  "disparity-sum", "graph-cut"]

# %%
print(which_exp)
if which_exp==4:
    subset_fraction_size = [0.5, 0.6, 0.5]
elif which_exp==2:
    subset_fraction_size = [0.3, 0.5]
else:
    print("Error: Experiment not defined!")
    exit()

# %%
dir_ls = os.listdir(f"./permutation_subsets_{which_exp}")
for i, s in enumerate(dir_ls):
    dir_ls[i]=dir_ls[i].split(".")[0]

# %%
for order in permutations(func_list, which_exp):
    per_func_list = list(order)
    print("permutation",per_func_list)
    filename = ""

    for func in per_func_list:
        filename += func+"_"
    filename = filename[:-1]

    if filename in dir_ls:
        print(f"{filename} is already present in directory hence skipping!")
        continue;

    print(filename)

    with open(f"../../../milo-base/cifar10/dataframe.pkl", "rb") as f:
        df = pickle.load(f)
        
    groups = df.groupby('Label')
    dataframes = [group for _, group in groups]

    for i, df in enumerate(dataframes):
        df["Features"] = df["Features"].to_numpy()
        df["Index"] = df["Index"].to_numpy()

    list_indexes = []

    if which_exp==2:
        sz = 2
    elif which_exp==4:
        sz = 3

    for idx in range(sz):
        fraction_size = subset_fraction_size[idx]
        final_indexes = []
        for i, df in enumerate(dataframes):
            features = df["Features"].to_numpy()
            indexes = df["Index"].to_numpy()

            print("start", features.shape, indexes.shape)

            if which_exp==4:
                # can choose a different strategy 
                func = per_func_list[idx+i%2]
            elif which_exp==2:
                func = per_func_list[idx]

            if func=="facility-location":
                obj = FacilityLocationFunction(n=features.shape[0], data=features, separate_rep=False, mode="dense", metric="cosine")
            elif func=="disparity-min":
                obj = DisparityMinFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
            elif func=="disparity-sum":
                obj = DisparitySumFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
            elif func=="graph-cut":
                obj = GraphCutFunction(n=features.shape[0], data=features, mode="dense", metric="cosine", lambdaVal=0.45)
            else:
                raise Exception("Sorry, no submodlib function defined")
            
            S = obj.maximize(int(fraction_size*features.shape[0]), optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
            # S = obj.maximize(30-10*idx, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
            S = list(map(lambda tuple: tuple[0], S))

            print(type(S))
            indexes = indexes[S]
            features = features[S]

            final_indexes.extend(list(indexes))

            _df = pd.DataFrame()
            _df["Features"] = features
            _df["Index"] = indexes

            dataframes[i] = _df

            print("end", features.shape[0], indexes.shape[0])

        list_indexes.append(final_indexes)
        
    with open(f"permutation_subsets_{which_exp}/{filename}.pkl", "wb") as f:
        pickle.dump(list_indexes, f)

# %%
# if which_exp==1:
#     subset_fraction_size = [0.5, 0.6, 0.5]

#     for order in generate_permutations(func_list):
#         per_func_list = list(order)
#         filename = ""
#         for func in per_func_list:
#             filename += func+"_"
#         filename = filename[:-1]

#         with open(f"../../../milo-base/cifar10/dataframe.pkl", "rb") as f:
#             df = pickle.load(f)
            
#         groups = df.groupby('Label')
#         dataframes = [group for _, group in groups]

#         for i, df in enumerate(dataframes):
#             df["Features"] = df["Features"].to_numpy()
#             df["Index"] = df["Index"].to_numpy()


#         list_indexes = []
#         for idx in range(len(func_list)-1):
#             fraction_size = subset_fraction_size[idx]
#             final_indexes = []
#             for i, df in enumerate(dataframes):
#                 features = df["Features"].to_numpy()
#                 indexes = df["Index"].to_numpy()

#                 print("start", features.shape, indexes.shape, 30-10*idx, idx)

#                 # can choose a different strategy 
#                 func = per_func_list[idx+i%2]

#                 if func=="facility-location":
#                     obj = FacilityLocationFunction(n=features.shape[0], data=features, separate_rep=False, mode="dense", metric="cosine")
#                 elif func=="disparity-min":
#                     obj = DisparityMinFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
#                 elif func=="disparity-sum":
#                     obj = DisparitySumFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
#                 elif func=="graph-cut":
#                     obj = GraphCutFunction(n=features.shape[0], data=features, mode="dense", metric="cosine", lambdaVal=0.45)
#                 else:
#                     raise Exception("Sorry, no submodlib function defined")
                
#                 S = obj.maximize(int(fraction_size*features.shape[0]), optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
#                 # S = obj.maximize(30-10*idx, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
#                 S = list(map(lambda tuple: tuple[0], S))

#                 print(type(S))
#                 indexes = indexes[S]
#                 features = features[S]

#                 final_indexes.extend(list(indexes))

#                 _df = pd.DataFrame()
#                 _df["Features"] = features
#                 _df["Index"] = indexes

#                 dataframes[i] = _df

#                 print("end", features.shape[0], indexes.shape[0])

#             list_indexes.append(final_indexes)
#         print(list_indexes)
            
#         with open(f"permutation_subsets/{filename}.pkl", "wb") as f:
#             pickle.dump(list_indexes, f)

# %% [markdown]
# ## 2 seq

# %%
# for i in permutations(func_list):
#     print(i)

# %%
# if which_exp==2:
#     subset_fraction_size = [0.3, 0.5]

#     for order in get_all_ordered_pairs(func_list):
#         per_func_list = list(order)
#         filename = ""
#         for func in per_func_list:
#             filename += func+"_"
#         filename = filename[:-1]

#         with open(f"../../../milo-base/cifar10/dataframe.pkl", "rb") as f:
#             df = pickle.load(f)
            
#         groups = df.groupby('Label')
#         dataframes = [group for _, group in groups]

#         for i, df in enumerate(dataframes):
#             df["Features"] = df["Features"].to_numpy()
#             df["Index"] = df["Index"].to_numpy()


#         list_indexes = []
#         for idx in range(len(func_list)-1):
#             fraction_size = subset_fraction_size[idx]
#             final_indexes = []
#             for i, df in enumerate(dataframes):
#                 features = df["Features"].to_numpy()
#                 indexes = df["Index"].to_numpy()

#                 print("start", features.shape, indexes.shape, 30-10*idx, idx)

#                 # can choose a different strategy 
#                 func = func_list[idx+i%2]

#                 if func=="facility-location":
#                     obj = FacilityLocationFunction(n=features.shape[0], data=features, separate_rep=False, mode="dense", metric="cosine")
#                 elif func=="disparity-min":
#                     obj = DisparityMinFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
#                 elif func=="disparity-sum":
#                     obj = DisparitySumFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
#                 elif func=="graph-cut":
#                     obj = GraphCutFunction(n=features.shape[0], data=features, mode="dense", metric="cosine", lambdaVal=0.45)
#                 else:
#                     raise Exception("Sorry, no submodlib function defined")
                
#                 S = obj.maximize(int(fraction_size*features.shape[0]), optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
#                 # S = obj.maximize(30-10*idx, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
#                 S = list(map(lambda tuple: tuple[0], S))

#                 print(type(S))
#                 indexes = indexes[S]
#                 features = features[S]

#                 final_indexes.extend(list(indexes))

#                 _df = pd.DataFrame()
#                 _df["Features"] = features
#                 _df["Index"] = indexes

#                 dataframes[i] = _df

#                 print("end", features.shape[0], indexes.shape[0])

#             list_indexes.append(final_indexes)
#         print(list_indexes)
            
#         with open(f"permutation_subsets/{filename}.pkl", "wb") as f:
#             pickle.dump(list_indexes, f)

# 2061183