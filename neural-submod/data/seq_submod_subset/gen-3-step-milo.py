# %%
import math
import numpy as np
from submodlib import FacilityLocationFunction, GraphCutFunction, DisparityMinFunction, DisparitySumFunction
import pickle
from itertools import permutations, combinations
from torchvision import datasets
import pandas as pd
from tqdm import tqdm 
import os

# %%
metric = "cosine"
func_list = ["facility-location", "disparity-min",  "disparity-sum", "graph-cut"]
# func_orders = [
#     ["fl", "gc", "fl", "dm", "dm", "ds"],
#     ["fl", "gc", "dm", "ds", "fl", "dm"],
#     ["fl", "dm", "fl", "gc", "dm", "ds"],
#     ["fl", "dm", "dm", "ds", "fl", "gc"],
#     ["dm", "ds", "fl", "dm", "fl", "gc"],
#     ["dm", "ds", "fl", "gc", "fl", "dm"],
# ]

func_orders = [
    ['fl', 'fl', 'dm'],
    ['gc', 'dm', 'ds'],
    ['fl', 'dm', 'fl'],
    ['gc', 'ds', 'dm'],

    ['dm', 'gc', 'ds'],
    
    ['dm', 'ds', 'gc'],
    ['dm', 'fl', 'fl'],
    ['ds', 'dm', 'gc'],

    ['ds', 'gc', 'dm']
]

print("lets start another round of bakchodi!!!")
frac_sz = [0.3, 0.5, 0.67]

dir_ls = os.listdir(f"./3-step-milo")
for i, s in enumerate(dir_ls):
    dir_ls[i]=dir_ls[i].split(".")[0]

#%%
for order in func_orders:
    with open(f"../../../milo-base/cifar10/dataframe.pkl", "rb") as f:
        df = pickle.load(f)
            
    groups = df.groupby('Label')
    dataframes = [group for _, group in groups]

    for i, df in enumerate(dataframes):
        df["Features"] = df["Features"].to_numpy()
        df["Index"] = df["Index"].to_numpy()

    filename = ""
    for func in order:
        filename += func+"_"
    filename = filename[:-1]

    # if filename in dir_ls:
    #     print(f"{filename} is already generated!!")
    #     continue
    
    print(f"{filename} data generating!!")

    list_indexes = []

    for idx, fraction_size in tqdm(enumerate(frac_sz)):
        final_indexes = []
        func = order[idx]

        for i, df in enumerate(dataframes):
            features = df["Features"].to_numpy()
            indexes = df["Index"].to_numpy()

            if func=="fl":
                obj = FacilityLocationFunction(n=features.shape[0], data=features, separate_rep=False, mode="dense", metric="cosine")
            elif func=="dm":
                obj = DisparityMinFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
            elif func=="ds":
                obj = DisparitySumFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
            elif func=="gc":
                obj = GraphCutFunction(n=features.shape[0], data=features, mode="dense", metric="cosine", lambdaVal=0.45)
            else:
                raise Exception("Sorry, no submodlib function defined!!!")

            S = obj.maximize(int(fraction_size*features.shape[0]), optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
            S = list(map(lambda tuple: tuple[0], S))

            indexes = indexes[S]
            features = features[S]

            final_indexes.extend(list(indexes))

            _df = pd.DataFrame()
            _df["Features"] = features
            _df["Index"] = indexes

            dataframes[i] = _df
            print("end", features.shape[0], indexes.shape[0])

        list_indexes.append(final_indexes)
        
    with open(f"3-step-milo/{filename}.pkl", "wb") as f:
        pickle.dump(list_indexes, f)

# 2420492
# mysession