# %%
import random
from tqdm import tqdm
import math
import numpy as np
from submodlib import FacilityLocationFunction, GraphCutFunction, DisparityMinFunction, DisparitySumFunction
import pickle
import time
import os
import timeit
import pandas as pd

# %%
def SGE_optimised(D, k, objFL, ϵ=1e-2, n=10, test=False):
    """Samples n subsets using stochastic-greedy.
    Args:
        D: number of training example
        k: The subset size.
        objFL: submodular function for conditional gain calculation
        ϵ: The error tolerance.
        n: The number of subsets to sample.

    Returns:
        A list of n subsets.
    """
    
    # Initialize empty subsets
    S = [set() for _ in range(n)]
    # Set random subset size for the stochastic-greedy algorithm
    s = int(D * math.log(1 / ϵ) / k)
    for i in tqdm(range(n)):
        for j in range(k):
            # Sample a random subset by sampling s elements from D \ Si
            R = random.choices(list(set(range(D)) - S[i]), k=s)
            # Use map to calculate marginal gains for all values in R with the custom set
            marginal_gains = list(map(lambda r: objFL.marginalGainWithMemoization(S[i], r), R))
            max_index = np.argmax(marginal_gains)
            max_r = R[max_index]
            objFL.updateMemoization(S[i], max_r)
            S[i].add(max_r)
            if test:
                print(R)
                print(marginal_gains)
                print(max_index, max_r)
                return S
        objFL.clearMemoization()
    return S

dataset = "cifar10"
metric = "cosine"


# %%
with open(f"./cifar10-dino-cls/dataframe-train.pkl", "rb") as f:
    df = pickle.load(f)

# %%
groups = df.groupby('Label')
dataframes = [group for _, group in groups]

# %%
print(df.columns)

# %% [markdown]
# # Generate milo subsets

# %%
from tqdm import tqdm

# %%
lambdaVal = 0.4

funcs = ["facility-location", "disparity-min",  "disparity-sum", "graph-cut"]
fracs = [0.3, 0.15, 0.1, 0.05, 0.5]

run_time = {}
for func in funcs:
    run_time[func] = {}
    for subset_size_fraction in fracs:
        time_to_get_subsets = 0
        num_sets = 100
        for i, df in tqdm(enumerate(dataframes)):
            features = df["Features"].to_numpy()
            
            if func=="facility-location":
                obj = FacilityLocationFunction(n=features.shape[0], data=features, separate_rep=False, mode="dense", metric="cosine")
            elif func=="disparity-min":
                obj = DisparityMinFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
            elif func=="disparity-sum":
                obj = DisparitySumFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
            elif func=="graph-cut":
                obj = GraphCutFunction(n=features.shape[0], data=features, mode="dense", metric="cosine", lambdaVal=lambdaVal)
            else:
                raise Exception("Sorry, no submodlib function defined")

            start_time = time.time()
            S = SGE_optimised(features.shape[0], int(features.shape[0]*subset_size_fraction), obj, n=num_sets)
            time_to_get_subsets += time.time() - start_time

            set_indexes = []
            for j in range(num_sets):
                set_indexes.append(set(df.iloc[list(S[j])].Index.tolist()))
            
            # path = f"./{dataset}/SGE-{metric}-2/{func}/class-data-{subset_size_fraction}-{num_sets}"
            path = f"./cifar10-dino-cls/SGE-{metric}/{func}/class-data-{subset_size_fraction}"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(f"{path}/class_{i}.pkl", "wb") as f:
                pickle.dump(set_indexes, f)
                
            # list_of_class_wise_subsets.append(set_indexes)

        if func=="graph-cut":
                run_time[func][f"{fracs}-{lambdaVal}"] = time_to_get_subsets 
        else:
            run_time[func][f"{fracs}"] = time_to_get_subsets 