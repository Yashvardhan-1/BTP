from typing import Set, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import random

class SetFunction(nn.Module):
    def __init__(self):
        pass

    def evaluate(self, X: Set[int]) -> float:
        return self.evaluate(X)

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        return self.evaluate_with_memoization(X)

    def marginal_gain(self, X: Set[int], item: int) -> float:
        return self.marginal_gain(X, item)

    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
       return self.marginal_gain_with_memoization(X, item)

    def update_memoization(self, X: Set[int], item: int) -> None:
        return self.update_memoization(X, item)


    def get_effective_ground_set(self) -> Set[int]:
        return self.get_effective_ground_set()

    def maximize(self, optimizer: str, budget: float, stopIfZeroGain: bool, stopIfNegativeGain: bool, verbose: bool,
                  costs: List[float] = None, cost_sensitive_greedy: bool = False, show_progress: bool = False, epsilon: float = 0.0) -> List[Tuple[int, float]]:
        optimizer = self._get_optimizer(optimizer)
        if optimizer:
            return optimizer.maximize(self, budget, stopIfZeroGain, stopIfZeroGain, verbose, show_progress, costs, cost_sensitive_greedy)
        else:
            print("Invalid Optimizer")
            return []

    def _get_optimizer(self, optimizer_name: str):
        if optimizer_name == "NaiveGreedy":
            return NaiveGreedyOptimizer()
        # define all optimizer classed into files
        elif optimizer_name == "LazyGreedy":
            return LazyGreedyOptimizer()
        elif optimizer_name == "StochasticGreedy":
            return StochasticGreedyOptimizer()
        elif optimizer_name == "LazierThanLazyGreedy":
            return LazierThanLazyGreedyOptimizer()
        else:
            return None

    def cluster_init(self, n: int, k_dense: List[List[float]], ground: Set[int],
                     partial: bool, lambda_: float) -> None:
        self.cluster_init(n, k_dense, ground, partial, lambda_)

    def set_memoization(self, X: Set[int]) -> None:
        self.set_memoization(X)

    def clear_memoization(self) -> None:
        self.clear_memoization()

class NaiveGreedyOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def equals(val1, val2, eps):
        return abs(val1 - val2) < eps

    def maximize(self, f_obj, budget, stop_if_zero_gain, stopIfNegativeGain, verbose, show_progress, costs, cost_sensitive_greedy):
        greedy_vector = []
        greedy_set = set()
        rem_budget = budget
        ground_set = f_obj.get_effective_ground_set()
        
        if verbose:
            print("Ground set:")
            print(ground_set)
            print(f"Num elements in groundset = {len(ground_set)}")
            print("Costs:")
            print(costs)
            print(f"Cost sensitive greedy: {cost_sensitive_greedy}")
            print("Starting the naive greedy algorithm")
            print("Initial greedy set:")
            print(greedy_set)

        f_obj.clear_memoization()
        best_id = None
        best_val = None
        step = 1
        display_next = step
        percent = 0
        N = rem_budget
        iter_count = 0

        while rem_budget > 0:
            best_id = None
            best_val = float("-inf")

            for i in ground_set:
                if i in greedy_set:
                    continue
                gain = f_obj.marginal_gain_with_memoization(greedy_set, i, False)
                
                if verbose:
                    print(f"Gain of {i} is {gain}")

                if gain > best_val:
                    best_id = i
                    best_val = gain

            if verbose:
                print(f"Next best item to add is {best_id} and its value addition is {best_val}")

            if (best_val < 0 and stopIfNegativeGain) or (
                self.equals(best_val, 0, 1e-5) and stop_if_zero_gain
            ):
                break
            else:
                f_obj.update_memoization(greedy_set, best_id)
                greedy_set.add(best_id)
                greedy_vector.append((best_id, best_val))
                rem_budget -= 1
                

                if verbose:
                    print(f"Added element {best_id} and the gain is {best_val}")
                    print(f"Updated greedy set: {greedy_set}")

                if show_progress:
                    percent = int((iter_count + 1.0) / N * 100)

                    if percent >= display_next:
                        print(
                            f"\r[{'|' * (percent // 5)}{' ' * (100 // 5 - percent // 5)}]",
                            end="",
                        )
                        print(f"{percent}% [Iteration {iter_count + 1} of {N}]", end="")
                        display_next += step

                    iter_count += 1

        return greedy_vector


def create_kernel(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None, method="sklearn", batch=0):

    if X_rep is not None:
        assert X_rep.shape[1] == X.shape[1]

    if mode == "dense":
        dense = None
        dense = globals()['create_kernel_dense_'+method](X, metric, X_rep, batch)
        return dense.clone().detach()

    elif mode == "sparse":
        if X_rep is not None:
            raise Exception("Sparse mode is not supported for separate X_rep")
        return create_sparse_kernel(X, metric, num_neigh, n_jobs, method)

    else:
        raise Exception("ERROR: unsupported mode")


def create_kernel_dense_sklearn(X, metric, X_rep=None, batch=0):
    dense = None
    D = None
    batch_size = batch
    if metric == "euclidean":
        if X_rep is None:
            # print(X.shape)
            # Process data in batches for torch.cdist
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device="cuda")
                # print(X_batch.shape)
                D_batch = torch.cdist(X_batch, X, p=2).to(device="cuda")
                gamma = 1 / X.shape[1]
                dense_batch = torch.exp(-D_batch * gamma).to(device="cuda")
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])
        else:
            # Process data in batches for torch.cdist
            for i in range(0, len(X_rep), batch_size):
                X_rep_batch = X_rep[i:i+batch_size].to(device="cuda")
                D_batch = torch.cdist(X_rep_batch, X).to(device="cuda")
                gamma = 1 / X.shape[1]
                dense_batch = torch.exp(-D_batch * gamma).to(device="cuda")
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])

    elif metric == "cosine":
        if X_rep is None:
            # Process data in batches for torch.nn.functional.cosine_similarity
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device="cuda")
                dense_batch = torch.nn.functional.cosine_similarity(X_batch.unsqueeze(1), X.unsqueeze(0), dim=2)
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])
        else:
            # Process data in batches for torch.nn.functional.cosine_similarity
            for i in range(0, len(X_rep), batch_size):
                X_rep_batch = X_rep[i:i+batch_size].to(device="cuda")
                dense_batch = torch.nn.functional.cosine_similarity(X_rep_batch, X, dim=1)
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])

    elif metric == "dot":
        if X_rep is None:
            # Process data in batches for torch.matmul
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device="cuda")
                dense_batch = torch.matmul(X_batch, X.t())
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])
        else:
            # Process data in batches for torch.matmul
            for i in range(0, len(X_rep), batch_size):
                X_rep_batch = X_rep[i:i+batch_size].to(device="cuda")
                dense_batch = torch.matmul(X_rep_batch, X.t())
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])

    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")

    if X_rep is not None:
        assert dense.shape == (X_rep.shape[0], X.shape[0])
    else:
        assert dense.shape == (X.shape[0], X.shape[0])

    torch.cuda.empty_cache()
    return dense

Vector = List[float]
Matrix = List[Vector]
Set = List[int]  # Considering integer elements for simplicity

# Euclidean similarity function
def euclidean_similarity(X1, X2):
    return torch.cdist(X1.unsqueeze(0), X2.unsqueeze(0), p=2).squeeze(0)


# Cosine similarity function
def cosine_similarity(a, b) -> float:
    dot_product = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

# Dot product function
def dot_prod(a, b) -> float:
    return torch.dot(a, b)

# Create kernel function for non-square kernel
def create_kernel_NS(X_ground, X_master, metric: str = "euclidean", batch = 0):
    print("NS started")
    n_ground = len(X_ground)
    n_master = len(X_master)
    k_dense = torch.zeros(n_master, n_ground)
    print("n_master",n_master)
    batch_size = batch
    for r in range(0, n_master, batch_size):
        #print(r)
        X_master_batch = X_master[r:r+batch_size]
        for c in range(0, n_ground, batch_size):
            X_ground_batch = X_ground[c:c+batch_size]
            if metric == "euclidean":
                sim_batch = euclidean_similarity(X_master_batch, X_ground_batch)
            elif metric == "cosine":
                sim_batch = cosine_similarity(X_master_batch, X_ground_batch)
            elif metric == "dot":
                sim_batch = dot_prod(X_master_batch, X_ground_batch)
            else:
                raise ValueError("Unsupported metric for kernel computation in Python")
            k_dense[r:r+sim_batch.size(0), c:c+sim_batch.size(1)] = sim_batch

    return k_dense


import math
from collections import defaultdict
import scipy

class LogDeterminant(SetFunction):

    def dot_product(self, x, y):
        return sum(xi * yi for xi, yi in zip(x, y))


    def __init__(self, n, mode, lambdaVal, arr_val=None, arr_count=None, arr_col=None, dense_kernel=None,batch_size=0, partial=None,
                  sijs=None, data=None, metric="cosine", num_neighbors=None, memoizedC = None, memoizedD = None, data_master = None):
        super(LogDeterminant, self).__init__()
        self.n = n
        self.mode = mode
        self.metric = metric
        self.sijs = sijs
        self.data = data
        self.num_neighbors = num_neighbors
        self.lambdaVal = lambdaVal
        self.sijs = None
        self.content = None
        self.effective_ground = None
        self.partial = partial
        self.effective_ground_set = set(range(n))
        self.memoizedC = memoizedC
        self.memoizedD = memoizedD
        self.data_master = data_master
        self.dense_kernel = dense_kernel
        self.batch_size=batch_size

        if self.n <= 0:
          raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse', 'clustered']:
          raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")

        if self.metric not in ['euclidean', 'cosine']:
        	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")
        if type(self.sijs) != type(None): # User has provided similarity kernel
          if type(self.sijs) == scipy.sparse.csr.csr_matrix:
            if num_neighbors is None or num_neighbors <= 0:
              raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
            if mode != "sparse":
              raise Exception("ERROR: Sparse kernel provided, but mode is not sparse")
          elif type(self.sijs) == np.ndarray:
            if mode != "dense":
              raise Exception("ERROR: Dense kernel provided, but mode is not dense")
          else:
            raise Exception("Invalid kernel provided")
          #TODO: is the below dimensionality check valid for both dense and sparse kernels?
          if np.shape(self.sijs)[0]!=self.n or np.shape(self.sijs)[1]!=self.n:
            raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity kernel")
          if type(self.data) != type(None):
            print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
        else: #similarity kernel has not been provided
          if type(self.data) != type(None):
            if np.shape(self.data)[0]!=self.n:
              raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")

            if self.mode == "dense":
              if self.num_neighbors  is not None:
                raise Exception("num_neighbors wrongly provided for dense mode")
              self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode

            #control
            print("check 1")
            #self.data = map(np.ndarray.tolist,self.data)
            #self.data = list(self.data)

            #print(self.num_neighbors)
            #self.cpp_content = np.array(create_kernel(X = torch.tensor(self.data,device="cuda"), metric = self.metric, num_neigh = self.num_neighbors, mode = self.mode).to_dense())

            y = torch.tensor(self.data)
            z = y.to(device="cuda")
            x= create_kernel(X=z, metric=self.metric, num_neigh=self.num_neighbors, mode=self.mode, batch=self.batch_size)
            # print("type x",type(x))
            #self.cpp_content = np.array(create_kernel(X=torch.tensor(self.data, device="cuda").cpu(), metric=self.metric, num_neigh=self.num_neighbors, mode=self.mode).to_dense())
            #val = self.cpp_content[0]
            val = x[0].cpu().detach().numpy()
            #print("val type", val.shape)
            row = list(x[1].cpu().detach().numpy().astype(int))
            #row = list(self.cpp_content[1].astype(int))
            #print("row type", type(row))
            col = list(x[2].cpu().detach().numpy().astype(int))
            if self.mode=="dense":
              self.sijs = np.zeros((n,n))
              self.sijs[row,col] = val
            if self.mode=="sparse":
              self.sijs = sparse.csr_matrix((val, (row, col)), [n,n])
          else:
            raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")


        #Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
        if self.mode=="dense":
          self.sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure

          if type(self.sijs[0])==int or type(self.sijs[0])==float: #Its critical that we pass a list of list to pybind11
                                          #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
            l=[]
            l.append(self.sijs)
            self.sijs=l

        self.effective_ground = self.get_effective_ground_set()
        if self.mode == 'dense':
          if self.dense_kernel == None:
             #control 2
             print("check 2")



            #print(self.num_neighbors)
            #self.cpp_content = np.array(create_kernel(X = torch.tensor(self.data,device="cuda"), metric = self.metric, num_neigh = self.num_neighbors, mode = self.mode).to_dense())

             y = torch.tensor(self.data)
             z = y.to(device="cuda")

             self.dense_kernel = create_kernel_NS(X_ground = z, X_master = z, metric = self.metric, batch = self.batch_size)
          if self.partial:
            self.effectiveGroundSet = self.data
          else:
            self.effectiveGroundSet = set(range(n))
            self.numEffectiveGroundset = len(self.effectiveGroundSet)
            self.memoizedC = [[] for _ in range(self.numEffectiveGroundset)]
            self.prevDetVal = 0
            self.memoizedD = []
            self.prevItem = -1

            if self.partial:
                ind = 0
                for it in self.effectiveGroundSet:
                    self.originalToPartialIndexMap[it] = ind
                    ind += 1
                    self.memoizedD.append(np.sqrt(self.dense_kernel[it][it] + self.lambdaVal))
            else:
                for i in range(self.n):
                    self.memoizedD.append(np.sqrt(self.dense_kernel[i][i] + self.lambdaVal))

        elif arr_val is not None and arr_count is not None and arr_col is not None:
            self.n = n
            self.mode = 'sparse'
            self.lambdaVal = lambdaVal
            self.sparseKernel = SparseSim(arr_val, arr_count, arr_col)
            self.effectiveGroundSet = set(range(n_))
            self.numEffectiveGroundset = len(self.effectiveGroundSet)
            self.memoizedC = [[] for _ in range(n_)]
            self.memoizedD = []
            self.prevDetVal = 0
            self.prevItem = -1

            for i in range(self.n):
                self.memoizedD.append(np.sqrt(self.sparseKernel.get_val(i, i) + self.lambdaVal))

        else:
            raise ValueError("Invalid constructor arguments. Please provide either denseKernel or sparse kernel data.")

    def evaluate(self, X):
        currMemoizedC = self.memoizedC.copy()
        currMemoizedD = self.memoizedD.copy()
        currprevItem = self.prevItem
        currprevDetVal = self.prevDetVal
        self.setMemoization(X)
        result = self.evaluate_with_memoization(X)
        self.memoizedC = currMemoizedC
        self.memoizedD = currMemoizedD
        self.prevItem = currprevItem
        self.prevDetVal = currprevDetVal
        return result

    def evaluate_with_memoization(self, X):
        return self.prevDetVal

    def marginal_gain(self, X, item):
        currMemoizedC = self.memoizedC.copy()
        currMemoizedD = self.memoizedD.copy()
        currprevItem = self.prevItem
        currprevDetVal = self.prevDetVal
        self.set_memoization(X)
        result = self.marginal_gain_with_memoization(X, item)
        self.memoizedC = currMemoizedC
        self.memoizedD = currMemoizedD
        self.prevItem = currprevItem
        self.prevDetVal = currprevDetVal
        return result

    def marginal_gain_with_memoization(self, X, item, enableChecks=True):
        effectiveX = X.intersection(self.effective_ground_set) if self.partial else X
        gain = 0

        # Check if the item is in X or the effective ground set
        if enableChecks:
            if item in effectiveX or (self.partial and item not in self.effective_ground_set):
                return 0

        itemIndex = self.originalToPartialIndexMap[item] if self.partial else item

        if self.mode == "dense":
            if len(effectiveX) == 0:
                gain = math.log(max(abs(self.memoizedD[itemIndex]) ** 2, 1e-12))
            elif len(effectiveX) == 1:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = self.dense_kernel[self.prevItem][item] / self.memoizedD[prevItemIndex]
                gain = math.log(max(abs(self.memoizedD[itemIndex]) ** 2 - e ** 2, 1e-12))
            else:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = (self.dense_kernel[self.prevItem][item] - self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[itemIndex])) / self.memoizedD[prevItemIndex]
                gain = math.log(max(abs(self.memoizedD[itemIndex]) ** 2 - e ** 2, 1e-12))
        elif self.mode == "sparse":
            if len(effectiveX) == 0:
                gain = math.log(max(abs(self.memoizedD[itemIndex]) ** 2, 1e-12))
            elif len(effectiveX) == 1:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = self.sparseKernel.get_val(self.prevItem, item) / self.memoizedD[prevItemIndex]
                gain = math.log(max(abs(self.memoizedD[itemIndex]) ** 2 - e ** 2, 1e-12))
            else:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = (self.sparseKernel.get_val(self.prevItem, item) - self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[itemIndex])) / self.memoizedD[prevItemIndex]
                gain = math.log(max(abs(self.memoizedD[itemIndex]) ** 2 - e ** 2, 1e-12))
        else:
            raise ValueError("Only dense and sparse mode supported")

        return gain



    def marginal_gains_with_memoization_vector(self, X, R, enableChecks=True):
        effectiveX = X.intersection(self.effective_ground_set) if self.partial else X

        # Check if the item is in X or the effective ground set
        if enableChecks:
            effective_indices = [self.originalToPartialIndexMap[item] if self.partial else item for item in R]
            mask = np.in1d(R, effectiveX)
            mask |= np.in1d(R, self.effective_ground_set) if self.partial else False
            gains = np.zeros(len(R))
            gains[mask] = 0
        else:
            effective_indices = [self.originalToPartialIndexMap[item] if self.partial else item for item in R]

        if self.mode == "dense":
            # Compute gains for dense mode
            if len(effectiveX) == 0:
                gains[~mask] = np.log(np.maximum(np.abs(self.memoizedD[effective_indices]) ** 2, 1e-12))
            elif len(effectiveX) == 1:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = self.dense_kernel[self.prevItem][R] / self.memoizedD[prevItemIndex]
                gains[~mask] = np.log(np.maximum(np.abs(self.memoizedD[effective_indices]) ** 2 - e ** 2, 1e-12))
            else:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = (self.dense_kernel[self.prevItem][R] - self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[effective_indices])) / self.memoizedD[prevItemIndex]
                gains[~mask] = np.log(np.maximum(np.abs(self.memoizedD[effective_indices]) ** 2 - e ** 2, 1e-12))
        elif self.mode == "sparse":
            # Compute gains for sparse mode
            if len(effectiveX) == 0:
                gains[~mask] = np.log(np.maximum(np.abs(self.memoizedD[effective_indices]) ** 2, 1e-12))
            elif len(effectiveX) == 1:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = self.sparseKernel.get_val(self.prevItem, R) / self.memoizedD[prevItemIndex]
                gains[~mask] = np.log(np.maximum(np.abs(self.memoizedD[effective_indices]) ** 2 - e ** 2, 1e-12))
            else:
                prevItemIndex = self.originalToPartialIndexMap.get(self.prevItem, self.prevItem) if self.partial else self.prevItem
                e = (self.sparseKernel.get_val(self.prevItem, R) - self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[effective_indices])) / self.memoizedD[prevItemIndex]
                gains[~mask] = np.log(np.maximum(np.abs(self.memoizedD[effective_indices]) ** 2 - e ** 2, 1e-12))
        else:
            raise ValueError("Only dense and sparse mode supported")

        return gains.tolist()



    def update_memoization(self, X, item):
        effectiveX = X.intersection(self.effective_ground_set) if self.partial else X

        if item in effectiveX:
            return

        if item not in self.effective_ground_set:
            return

        self.prevDetVal += self.marginal_gain_with_memoization(X, item)

        if len(effectiveX) == 0:
            pass
        else:
            prevItemIndex = self.originalToPartialIndexMap[self.prevItem] if self.partial else self.prevItem
            prevDValue = self.memoizedD[prevItemIndex]

            for i in self.effectiveGroundSet:
                iIndex = self.originalToPartialIndexMap[i] if self.partial else i

                if i in effectiveX:
                    continue

                e = 0
                if len(effectiveX) == 1:
                    e = self.dense_kernel[self.prevItem][i] / prevDValue
                    self.memoizedC[iIndex].append(e)
                else:
                    e = (self.dense_kernel[self.prevItem][i] -
                         self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[iIndex])) / prevDValue
                    self.memoizedC[iIndex].append(e)

                self.memoizedD[iIndex] = math.sqrt(math.fabs(self.memoizedD[iIndex] * self.memoizedD[iIndex] - e * e))

        self.prevItem = item

    def get_effective_ground_set(self):
        return self.effective_ground_set

    def clear_memoization(self):
        self.memoizedC.clear()
        self.memoizedC = defaultdict(list)
        self.prevDetVal = 0
        self.prevItem = -1

        if self.mode == "dense":
            if self.partial:
                for it in self.effective_ground_set:
                    index = self.originalTo
