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


import torch
import random
from typing import List, Tuple, Set

class NaiveGreedyOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def equals(val1, val2, eps):
        return abs(val1 - val2) < eps

    def maximize(
        self, f_obj, budget, stop_if_zero_gain, stopIfNegativeGain, verbose, show_progress, costs, cost_sensitive_greedy
    ):
        greedy_vector = []
        greedy_set = set()
        if not costs:
            # greedy_vector = [None] * budget
            greedy_set = set()
        rem_budget = budget
        ground_set = f_obj.get_effective_ground_set()
        #print(ground_set)
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
                # print(gain)
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


import torch
import torch.nn.functional as F
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import pickle
import time
import os

def cos_sim_square(A):
    similarity = torch.matmul(A, A.t())

    square_mag = torch.diag(similarity)

    inv_square_mag = 1 / square_mag
    inv_square_mag[torch.isinf(inv_square_mag)] = 0

    inv_mag = torch.sqrt(inv_square_mag)

    cosine = similarity * inv_mag
    cosine = cosine.t() * inv_mag
    return cosine

def cos_sim_rectangle(A, B):
    num = torch.matmul(A, B.t())
    p1 = torch.sqrt(torch.sum(A**2, dim=1)).unsqueeze(1)
    p2 = torch.sqrt(torch.sum(B**2, dim=1)).unsqueeze(0)
    return num / (p1 * p2)

def create_sparse_kernel(X, metric, num_neigh, n_jobs=1, method="sklearn"):
    if num_neigh > X.shape[0]:
        raise Exception("ERROR: num of neighbors can't be more than the number of datapoints")
    dense = None
    dense = create_kernel_dense_sklearn(X, metric)
    dense_ = None
    if num_neigh == -1:
        num_neigh = X.shape[0]  # default is the total number of datapoints

    # Assuming X is a PyTorch tensor


    # Use PyTorch functions for the nearest neighbors search
    if metric == 'euclidean':
      distances = torch.cdist(X, X, p=2)  # Euclidean distance
    elif metric == 'cosine':
      distances = 1 - torch.nn.functional.cosine_similarity(X, X, dim=1)  # Cosine similarity as distance

    # Exclude the distance to oneself (diagonal elements)
    distances.fill_diagonal_(float('inf'))

    # Find the indices of the k-nearest neighbors using torch.topk
    _, ind = torch.topk(distances, k=num_neigh, largest=False)

    # ind_l = [(index[0], x.item()) for index, x in torch.ndenumerate(ind)]
        # Convert indices to row and col lists
    row = []
    col = []
    for i, indices_row in enumerate(ind):
        for j in indices_row:
            row.append(i)
            col.append(j.item())

    mat = torch.zeros_like(distances)
    mat[row, col] = 1
    dense_ = dense * mat  # Only retain similarity of nearest neighbors
    sparse_coo = torch.sparse_coo_tensor(torch.tensor([row, col]), mat[row, col], dense.size())
    # Convert the COO tensor to CSR format
    sparse_csr = sparse_coo.coalesce()
    return sparse_csr
    # pass


def create_kernel_dense(X, metric, method="sklearn"):
    dense = None
    if method == "sklearn":
        dense = create_kernel_dense_sklearn(X, metric)
    else:
        raise Exception("For creating dense kernel, only 'sklearn' method is supported")
    return dense

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


from typing import List, Set
import random

class GraphCutFunction(SetFunction):
    # def __init__(self, n: int, mode: str, metric: str, master_ground_kernel: List[List[float]] = None,
    #              ground_ground_kernel: List[List[float]] = None, arr_val: List[float] = None,
    #              arr_count: List[int] = None, arr_col: List[int] = None, partial: bool = False,
    #              ground: Set[int] = None, lambdaVal: float = 0.0):
    def __init__(self, n, mode, lambdaVal, separate_rep=None, n_rep=None, mgsijs=None, ggsijs=None, data=None, data_rep=None, metric="cosine", num_neighbors=None,batch_size=0,
                 master_ground_kernel: List[List[float]] = None,
                 ground_ground_kernel: List[List[float]] = None, arr_val: List[float] = None,
                 arr_count: List[int] = None, arr_col: List[int] = None, partial: bool = False,
                 ground: Set[int] = None):
        super(SetFunction, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n = n
        self.mode = mode
        self.lambda_ = lambdaVal
        self.separate_rep=separate_rep
        self.n_rep = n_rep
        self.partial = partial
        self.original_to_partial_index_map = {}
        self.mgsijs = mgsijs
        self.ggsijs = ggsijs
        self.data = data
        self.data_rep=data_rep
        self.metric = metric
        self.num_neighbors = num_neighbors
        self.effective_ground_set = set(range(n))
        self.clusters=None
        self.cluster_sijs=None
        self.cluster_map=None
        self.ggsijs = None
        self.mgsijs = None
        self.content = None
        self.effective_ground = None
        self.batch_size= batch_size

        if self.n <= 0:
          raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse']:
          raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")
        if self.separate_rep == True:
          if self.n_rep is None or self.n_rep <=0:
            raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")
          if self.mode != "dense":
            raise Exception("Only dense mode supported if separate_rep = True")
          if (type(self.mgsijs) != type(None)) and (type(self.mgsijs) != np.ndarray):
            raise Exception("mgsijs provided, but is not dense")
          if (type(self.ggsijs) != type(None)) and (type(self.ggsijs) != np.ndarray):
            raise Exception("ggsijs provided, but is not dense")

        if mode == "dense":
            self.master_ground_kernel = master_ground_kernel
            self.ground_ground_kernel = ground_ground_kernel

            if ground_ground_kernel is not None:
                self.separate_master = True

            if partial:
                self.effective_ground_set = ground
            else:
                self.effective_ground_set = set(range(n))

            self.num_effective_ground_set = len(self.effective_ground_set)

            self.n_master = self.num_effective_ground_set
            self.master_set = self.effective_ground_set

            if partial:
                self.original_to_partial_index_map = {elem: ind for ind, elem in enumerate(self.effective_ground_set)}

            self.total_similarity_with_subset = [random.random() for _ in range(self.num_effective_ground_set)]
            self.total_similarity_with_master = [random.random() for _ in range(self.num_effective_ground_set)]
            self.master_ground_kernel = [[random.random() for _ in range(self.num_effective_ground_set)] for _ in range(self.num_effective_ground_set)]
            self.ground_ground_kernel = [[random.random() for _ in range(self.num_effective_ground_set)] for _ in range(self.num_effective_ground_set)]
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if partial else elem
                self.total_similarity_with_subset[index] = 1
                self.total_similarity_with_master[index] = 1
                for j in self.master_set:
                    self.total_similarity_with_master[index] += self.master_ground_kernel[j][elem]

            if self.separate_rep == True:
              if type(self.mgsijs) == type(None):
                #not provided mgsij - make it
                if (type(data) == type(None)) or (type(data_rep) == type(None)):
                  raise Exception("Data missing to compute mgsijs")
                if np.shape(self.data)[0]!=self.n or np.shape(self.data_rep)[0]!=self.n_rep:
                  raise Exception("ERROR: Inconsistentcy between n, n_rep and no of examples in the given ground data matrix and represented data matrix")

                #create_kernel_NS is there .................... find it and define it not found in helper.py but used as here
                # self.mgsijs = np.array(subcp.create_kernel_NS(self.data.tolist(),self.data_rep.tolist(), self.metric))
              else:
                #provided mgsijs - verify it's dimensionality
                if np.shape(self.mgsijs)[1]!=self.n or np.shape(self.mgsijs)[0]!=self.n_rep:
                  raise Exception("ERROR: Inconsistency between n_rep, n and no of rows, columns of given mg kernel")
              if type(self.ggsijs) == type(None):
                #not provided ggsijs - make it
                if type(data) == type(None):
                  raise Exception("Data missing to compute ggsijs")
                if self.num_neighbors is not None:
                  raise Exception("num_neighbors wrongly provided for dense mode")
                self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
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
                self.ggsijs = np.zeros((n,n))
                self.ggsijs[row,col] = val
              else:
                #provided ggsijs - verify it's dimensionality
                if np.shape(self.ggsijs)[0]!=self.n or np.shape(self.ggsijs)[1]!=self.n:
                  raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity gg kernel")

            else:
              if (type(self.ggsijs) == type(None)) and (type(self.mgsijs) == type(None)):
                #no kernel is provided make ggsij kernel
                if type(data) == type(None):
                  raise Exception("Data missing to compute ggsijs")
                if self.num_neighbors is not None:
                  raise Exception("num_neighbors wrongly provided for dense mode")
                self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
                #self.data = map(np.ndarray.tolist,self.data)
                #self.data = list(self.data)

                #print(self.num_neighbors)
                #self.cpp_content = np.array(create_kernel(X = torch.tensor(self.data,device="cuda"), metric = self.metric, num_neigh = self.num_neighbors, mode = self.mode).to_dense())

                #y = torch.tensor(self.data)
                z = self.data.to(device="cuda")
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
                self.ggsijs = np.zeros((n,n))
                self.ggsijs[row,col] = val
              elif (type(self.ggsijs) == type(None)) and (type(self.mgsijs) != type(None)):
                #gg is not available, mg is - good
                #verify that it is dense and of correct dimension
                if (type(self.mgsijs) != np.ndarray) or np.shape(self.mgsijs)[1]!=self.n or np.shape(self.mgsijs)[0]!=self.n:
                  raise Exception("ERROR: Inconsistency between n and no of rows, columns of given kernel")
                self.ggsijs = self.mgsijs
              elif (type(self.ggsijs) != type(None)) and (type(self.mgsijs) == type(None)):
                #gg is available, mg is not - good
                #verify that it is dense and of correct dimension
                if (type(self.ggsijs) != np.ndarray) or np.shape(self.ggsijs)[1]!=self.n or np.shape(self.ggsijs)[0]!=self.n:
                  raise Exception("ERROR: Inconsistency between n and no of rows, columns of given kernel")
              else:
                #both are available - something is wrong
                raise Exception("Two kernels have been wrongly provided when separate_rep=False")
        elif mode == "sparse":
            if self.separate_rep == True:
                raise Exception("Separate represented is supported only in dense mode")
            if self.num_neighbors is None or self.num_neighbors <=0:
              raise Exception("Valid num_neighbors is needed for sparse mode")
            if (type(self.ggsijs) == type(None)) and (type(self.mgsijs) == type(None)):
              #no kernel is provided make ggsij sparse kernel
              if type(data) == type(None):
                raise Exception("Data missing to compute ggsijs")
              self.content = np.array(create_kernel(X = torch.tensor(self.data), metric = self.metric, num_neigh = self.num_neighbors).to_dense())
              val = self.content[0]
              row = list(self.content[1].astype(int))
              col = list(self.content[2].astype(int))
              self.ggsijs = sparse.csr_matrix((val, (row, col)), [n,n])
            elif (type(self.ggsijs) == type(None)) and (type(self.mgsijs) != type(None)):
              #gg is not available, mg is - good
              #verify that it is sparse
              if type(self.mgsijs) != scipy.sparse.csr.csr_matrix:
                raise Exception("Provided kernel is not sparse")
              self.ggsijs = self.mgsijs
            elif (type(self.ggsijs) != type(None)) and (type(self.mgsijs) == type(None)):
              #gg is available, mg is not - good
              #verify that it is dense and of correct dimension
              if type(self.ggsijs) != scipy.sparse.csr.csr_matrix:
                raise Exception("Provided kernel is not sparse")
            else:
              #both are available - something is wrong
              raise Exception("Two kernels have been wrongly provided when separate_rep=False")

        if self.separate_rep==None:
            self.separate_rep = False

        if self.mode=="dense" and self.separate_rep == False :
            self.ggsijs = self.ggsijs.tolist() #break numpy ndarray to native list of list datastructure

            if type(self.ggsijs[0])==int or type(self.ggsijs[0])==float: #Its critical that we pass a list of list to pybind11
                                            #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
              l=[]
              l.append(self.ggsijs)
              self.ggsijs=l

        elif self.mode=="dense" and self.separate_rep == True :
            self.ggsijs = self.ggsijs.tolist() #break numpy ndarray to native list of list datastructure

            if type(self.ggsijs[0])==int or type(self.ggsijs[0])==float: #Its critical that we pass a list of list to pybind11
                                            #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
              l=[]
              l.append(self.ggsijs)
              self.ggsijs=l

            self.mgsijs = self.mgsijs.tolist() #break numpy ndarray to native list of list datastructure

            if type(self.mgsijs[0])==int or type(self.mgsijs[0])==float: #Its critical that we pass a list of list to pybind11
                                            #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
              l=[]
              l.append(self.mgsijs)
              self.mgsijs=l

            # self.cpp_obj = GraphCutpy(self.n, self.cpp_mgsijs, self.cpp_ggsijs, self.lambdaVal)

        elif self.mode == "sparse":
            self.ggsijs = {}
            # self.ggsijs['arr_val'] = self.ggsijs.data.tolist() #contains non-zero values in matrix (row major traversal)
            # self.ggsijs['arr_count'] = self.ggsijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
            # self.ggsijs['arr_col'] = self.ggsijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
            # # self.cpp_obj = GraphCutpy(self.n, self.cpp_ggsijs['arr_val'], self.cpp_ggsijs['arr_count'], self.cpp_ggsijs['arr_col'], lambdaVal)
        else:
            raise Exception("Invalid")

        self.effective_ground = self.get_effective_ground_set()

        # if mode == "dense":

        # elif mode == "sparse":
        #     if not arr_val or not arr_count or not arr_col:
        #         raise ValueError("Error: Empty/Corrupt sparse similarity kernel")

        #     self.sparse_kernel = SparseSim(arr_val, arr_count, arr_col)

        #     self.effective_ground_set = set(range(n))
        #     self.num_effective_ground_set = len(self.effective_ground_set)

        #     self.n_master = self.num_effective_ground_set
        #     self.master_set = self.effective_ground_set

        #     self.total_similarity_with_subset = [0] * n
        #     self.total_similarity_with_master = [0] * n

        #     for i in range(n):
        #         self.total_similarity_with_subset[i] = 0
        #         self.total_similarity_with_master[i] = 0

        #         for j in range(n):
        #             self.total_similarity_with_master[i] += self.sparse_kernel.get_val(j, i)

        # else:
        #     raise ValueError("Invalid mode")

    def evaluate(self, X: Set[int]) -> float:
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if not effective_x:
            return 0

        result = 0

        if self.mode == "dense":
            for elem in effective_x:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                result += self.total_similarity_with_master[index]

                for elem2 in effective_x:
                    result -= self.lambda_ * self.ground_ground_kernel[elem][elem2]

        elif self.mode == "sparse":
            for elem in effective_x:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                result += self.total_similarity_with_master[index]

                for elem2 in effective_x:
                    result -= self.lambda_ * self.sparse_kernel.get_val(elem, elem2)

        return result

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if not effective_x:
            return 0

        result = 0

        if self.mode == "dense" or self.mode == "sparse":
            for elem in effective_x:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                result += self.total_similarity_with_master[index] - self.lambda_ * self.total_similarity_with_subset[index]

        return result

    def marginal_gain(self, X: Set[int], item: int) -> float:
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if item in effective_x or item not in self.effective_ground_set:
            return 0

        gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item]

        if self.mode == "dense":
            for elem in effective_x:
                gain -= 2 * self.lambda_ * self.ground_ground_kernel[item][elem]
            gain -= self.lambda_ * self.ground_ground_kernel[item][item]

        elif self.mode == "sparse":
            for elem in effective_x:
                gain -= 2 * self.lambda_ * self.sparse_kernel.get_val(item, elem)
            gain -= self.lambda_ * self.sparse_kernel.get_val(item, item)
        return gain

    # def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
    #     effective_x = X.intersection(self.effective_ground_set) if self.partial else X

    #     if enable_checks and item in effective_x:
    #         return 0

    #     if self.partial and item not in self.effective_ground_set:
    #         return 0

    #     gain = 0

    #     if self.mode == "dense":
    #         index = self.original_to_partial_index_map[item] if self.partial else item
    #         gain = self.total_similarity_with_master[index] - 2 * self.lambda_ * self.total_similarity_with_subset[index]
    #         gain = self.total_similarity_with_master[index] - 2 * self.lambda_ * self.total_similarity_with_subset[index] - self.lambda_ * self.ground_ground_kernel[item][item]

    #     elif self.mode == "sparse":
    #         index = self.original_to_partial_index_map[item] if self.partial else item
    #         gain = self.total_similarity_with_master[index] - 2 * self.lambda_ * self.total_similarity_with_subset[index] - self.lambda_ * self.sparse_kernel.get_val(item, item)

    #     return gain


    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
        effective_X = set()
        gain = 0
        if self.partial:
            effective_X = X.intersection(self.effective_ground_set)
        else:
            effective_X = X

        if enable_checks and item in effective_X:
            return 0

        if self.partial and item not in self.effective_ground_set:
            return 0

        if self.mode == 'dense':
            gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item] \
                  - 2 * self.lambda_ * self.total_similarity_with_subset[self.original_to_partial_index_map[item] if self.partial else item] \
                  - self.lambda_ * self.ground_ground_kernel[item][item]
        elif self.mode == 'sparse':
            gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item] \
                  - 2 * self.lambda_ * self.total_similarity_with_subset[self.original_to_partial_index_map[item] if self.partial else item] \
                  - self.lambda_ * self.sparse_kernel.get_val(item, item)
        else:
            raise ValueError("Error: Only dense and sparse mode supported")
        # print("gain value",gain)
        return gain


    def marginal_gain_with_memoization_vector(self, X: Set[int], R: List[int], enable_checks: bool = True) -> List[float]:
        effective_X = set()
        gains = []

        if self.partial:
            effective_X = X.intersection(self.effective_ground_set)
        else:
            effective_X = X

        for item in R:
            gain = 0

            if enable_checks and item in effective_X:
                gains.append(0)
                continue

            if self.partial and item not in self.effective_ground_set:
                gains.append(0)
                continue

            if self.mode == 'dense':
                gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item] \
                    - 2 * self.lambda_ * self.total_similarity_with_subset[self.original_to_partial_index_map[item] if self.partial else item] \
                    - self.lambda_ * self.ground_ground_kernel[item][item]
            elif self.mode == 'sparse':
                gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item] \
                    - 2 * self.lambda_ * self.total_similarity_with_subset[self.original_to_partial_index_map[item] if self.partial else item] \
                    - self.lambda_ * self.sparse_kernel.get_val(item, item)
            else:
                raise ValueError("Error: Only dense and sparse mode supported")

            gains.append(gain)

        return gains


    def update_memoization(self, X: Set[int], item: int):
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if item in effective_x or item not in self.effective_ground_set:
            return

        if self.mode == "dense":
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                # self.total_similarity_with_subset[index] += self.ground_ground_kernel[elem][item]

        elif self.mode == "sparse":
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                self.total_similarity_with_subset[index] += self.sparse_kernel.get_val(elem, item)

    def get_effective_ground_set(self) -> Set[int]:
        return self.effective_ground_set

    def clear_memoization(self):
        if self.mode == "dense" or self.mode == "sparse":
            self.total_similarity_with_subset = [0] * self.num_effective_ground_set

    def set_memoization(self, X: Set[int]):
        temp = set()
        for elem in X:
            self.update_memoization(temp, elem)
            temp.add(elem)