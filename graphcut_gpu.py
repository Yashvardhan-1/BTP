from typing import Set, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import pickle
import time
import os


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


import torch

def create_kernel_dense_sklearn(X, metric, X_rep=None, batch_size=256):
    if metric not in ["euclidean", "cosine", "dot"]:
        raise ValueError("ERROR: unsupported metric for this method of kernel creation")

    # Initialize an empty list to hold batch results
    batched_results = []
    device = X.device

    # Pre-compute norms for cosine similarity
    if metric == "cosine":
        X_norm = X / X.norm(dim=1, keepdim=True)

    if X_rep is not None:
        X_rep = X_rep.to(device)
        if metric == "cosine":
            X_rep = X_rep / X_rep.norm(dim=1, keepdim=True)

    # Determine the number of batches
    total_samples = X.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        # Slice the batch for X
        X_batch = X[start_idx:end_idx].to(device)

        if metric == "euclidean":
            if X_rep is None:
                D_batch = torch.cdist(X_batch, X, p=2)
            else:
                D_batch = torch.cdist(X_batch, X_rep, p=2)
            gamma = 1 / X.shape[1]
            dense_batch = torch.exp(-D_batch * gamma)

        elif metric == "cosine":
            X_batch_norm = X_batch / X_batch.norm(dim=1, keepdim=True)
            if X_rep is None:
                dense_batch = torch.matmul(X_batch_norm, X_norm.t())
            else:
                dense_batch = torch.matmul(X_batch_norm, X_rep.t())

        elif metric == "dot":
            if X_rep is None:
                dense_batch = torch.matmul(X_batch, X.t())
            else:
                dense_batch = torch.matmul(X_batch, X_rep.t())

        batched_results.append(dense_batch)

    # Concatenate all batch results along the dimension 0
    dense = torch.cat(batched_results, dim=0)

    return dense


def create_sparse_kernel(X, metric, num_neigh, n_jobs=1, method="sklearn"):
    if num_neigh > X.shape[0]:
        raise Exception("ERROR: num of neighbors can't be more than the number of datapoints")
    dense = None
    dense = create_kernel_dense_sklearn(X, metric)
    dense_ = None
    if num_neigh == -1:
        num_neigh = X.shape[0]  # default is the total number of datapoints

    if metric == 'euclidean':
        distances = torch.cdist(X, X, p=2)  # Euclidean distance
    elif metric == 'cosine':
        distances = 1 - torch.nn.functional.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)

    # Exclude the distance to oneself (diagonal elements)
    distances.fill_diagonal_(float('inf'))

    # Find the indices of the k-nearest neighbors using torch.topk
    _, ind = torch.topk(distances, k=num_neigh, largest=False, dim=1)

    row, col = torch.meshgrid(torch.arange(ind.shape[0]), torch.arange(num_neigh), indexing='ij')
    row = row.reshape(-1)
    col = ind.reshape(-1)

    values = torch.ones_like(row, dtype=dense.dtype)  # Use ones to mark connections in the adjacency matrix
    sparse_coo = torch.sparse_coo_tensor(torch.stack((row, col)), values, dense.size())
    dense_ = dense * sparse_coo.to_dense()  # Element-wise multiplication to retain only neighbor connections

    # Return the result in sparse format
    return sparse_coo.coalesce()

def create_kernel(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None, method="sklearn",batch_size=0):
    if X_rep is not None:
        assert X_rep.shape[1] == X.shape[1]

    if mode == "dense":
        #print("here")
        dense = globals()['create_kernel_dense_'+method](X, metric, X_rep,batch_size)
        return dense

    elif mode == "sparse":
        if X_rep is not None:
            raise Exception("Sparse mode is not supported for separate X_rep")
        return create_sparse_kernel(X, metric, num_neigh, n_jobs, method)

    else:
        raise Exception("ERROR: unsupported mode")


from typing import List, Set
import random

from typing import List, Set
import random



class GraphCutpy(SetFunction):
    def __init__(self, n, mode, lambdaVal, separate_rep=None, n_rep=None, mgsijs=None, ggsijs=None, data=None, data_rep=None, metric="cosine", num_neighbors=None,
                 master_ground_kernel: List[List[float]] = None, ground_ground_kernel: List[List[float]] = None,
                 arr_val: List[float] = None, arr_count: List[int] = None, arr_col: List[int] = None,
                 partial: bool = False, ground: Set[int] = None, batch_size=0):
        super(GraphCutpy, self).__init__()
        self.n = n
        self.mode = mode
        self.lambda_ = lambdaVal
        self.separate_rep = separate_rep
        self.n_rep = n_rep
        self.partial = partial
        self.original_to_partial_index_map = {}
        self.mgsijs = mgsijs
        self.ggsijs = ggsijs
        self.data = data
        self.data_rep = data_rep
        self.metric = metric
        self.num_neighbors = num_neighbors
        self.effective_ground_set = set(range(n))
        self.clusters = None
        self.cluster_sijs = None
        self.cluster_map = None
        self.content = None
        self.effective_ground = None
        self.batch_size = batch_size
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        #print("inside")
        self.data = self.data.to(self.device)

        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse']:
            raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")

        if self.separate_rep:
            if self.n_rep is None or self.n_rep <= 0:
                raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")
            if self.mode != "dense":
                raise Exception("Only dense mode supported if separate_rep = True")
            if self.mgsijs is not None and not isinstance(self.mgsijs, torch.Tensor):
                raise Exception("mgsijs provided, but is not dense")
            if self.ggsijs is not None and not isinstance(self.ggsijs, torch.Tensor):
                raise Exception("ggsijs provided, but is not dense")

        if mode == "dense":
            #print("in dense")
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

            self.total_similarity_with_subset = torch.ones(self.num_effective_ground_set, device=self.device)
            self.total_similarity_with_master = torch.ones(self.num_effective_ground_set, device=self.device)
            self.master_ground_kernel = torch.rand(self.num_effective_ground_set, self.num_effective_ground_set, device=self.device)
            self.ground_ground_kernel = torch.rand(self.num_effective_ground_set, self.num_effective_ground_set, device=self.device)
            
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if partial else elem
                self.total_similarity_with_master[index] = torch.sum(self.master_ground_kernel[:, elem])

            if self.separate_rep:
                if self.mgsijs is None:
                    if self.data is None or self.data_rep is None:
                        raise Exception("Data missing to compute mgsijs")
                    if self.data.shape[0] != self.n or self.data_rep.shape[0] != self.n_rep:
                        raise Exception("ERROR: Inconsistency between n, n_rep and number of examples in the given ground data matrix and represented data matrix")

                    self.mgsijs = create_kernel_NS(self.data, self.data_rep, self.metric).to(self.device)
                else:
                    if self.mgsijs.shape[1] != self.n or self.mgsijs.shape[0] != self.n_rep:
                        raise Exception("ERROR: Inconsistency between n_rep, n and number of rows, columns of given mg kernel")

                if self.ggsijs is None:
                    if self.data is None:
                        raise Exception("Data missing to compute ggsijs")
                    if self.num_neighbors is not None:
                        raise Exception("num_neighbors wrongly provided for dense mode")
                    self.num_neighbors = self.data.shape[0]  # Using all data as num_neighbors in case of dense mode

                    self.ggsijs = create_kernel(self.data, self.metric, num_neigh=self.num_neighbors, batch_size=self.batch_size).to(self.device)
                else:
                    if self.ggsijs.shape[0] != self.n or self.ggsijs.shape[1] != self.n:
                        raise Exception("ERROR: Inconsistency between n and dimensionality of given similarity gg kernel")

            else:
                if self.ggsijs is None and self.mgsijs is None:
                    if self.data is None:
                        raise Exception("Data missing to compute ggsijs")
                    if self.num_neighbors is not None:
                        raise Exception("num_neighbors wrongly provided for dense mode")
                    self.num_neighbors = self.data.shape[0]  # Using all data as num_neighbors in case of dense mode
                    #print("create kernel")
                    self.ggsijs = create_kernel(self.data, self.metric, self.mode, self.num_neighbors, batch_size=self.batch_size).to(self.device)
                    #print("taking time?")
                elif self.ggsijs is None and self.mgsijs is not None:
                    if self.mgsijs.shape[1] != self.n or self.mgsijs.shape[0] != self.n:
                        raise Exception("ERROR: Inconsistency between n and number of rows, columns of given kernel")
                    self.ggsijs = self.mgsijs
                elif self.ggsijs is not None and self.mgsijs is None:
                    if self.ggsijs.shape[1] != self.n or self.ggsijs.shape[0] != self.n:
                        raise Exception("ERROR: Inconsistency between n and number of rows, columns of given kernel")
                else:
                    raise Exception("Two kernels have been wrongly provided when separate_rep=False")

        self.effective_ground = self.get_effective_ground_set()

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
                    result -= self.lambda_ * self.ground_ground_kernel[elem, elem2]

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

    def marginal_gain(self, X: Set[int], items: List[int]) -> torch.Tensor:
        #print("gain")

        effective_X = torch.tensor(list(X.intersection(self.effective_ground_set) if self.partial else X), device=self.device, dtype=torch.long)
        item_indices = torch.tensor([self.original_to_partial_index_map[item] if self.partial else item for item in items], device=self.device)
        
        # Initialize gains with the total similarity with the master set
        total_similarity = self.total_similarity_with_master[item_indices].clone()

        if effective_X.size(0) == 0:
            return total_similarity

        if self.mode == "dense":
            # Matrix multiplication to sum similarities
            effective_X_matrix = self.ground_ground_kernel[item_indices][:, effective_X]
            sum_similarities = torch.sum(effective_X_matrix, dim=1)

            # Subtract the weighted sum of similarities
            total_similarity -= 2 * self.lambda_ * sum_similarities

            # Subtract self-similarity terms
            self_similarities = self.ground_ground_kernel[item_indices, item_indices]
            total_similarity -= self.lambda_ * self_similarities

            gains = total_similarity

            #print("out")


        elif self.mode == "sparse":
            item_indices = torch.tensor([self.original_to_partial_index_map[item] if self.partial else item for item in items], device=device)
            total_similarity = self.total_similarity_with_master[item_indices]

            if len(effective_X) > 0:
                effective_X_matrix = self.sparse_kernel.get_val(items, effective_X)
                sum_similarities = effective_X_matrix.sum(dim=1)
                total_similarity -= 2 * self.lambda_ * sum_similarities

            self_similarities = self.sparse_kernel.get_val(items, items)
            total_similarity -= self.lambda_ * self_similarities
            gains = total_similarity

        return gains

    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool) -> float:
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
        return gain

    # def update_memoization(self, X: Set[int], item: int):
    #     effective_x = X.intersection(self.effective_ground_set) if self.partial else X

    #     if item in effective_x or item not in self.effective_ground_set:
    #         return

    #     if self.mode == "dense":
    #         for elem in self.effective_ground_set:
    #             index = self.original_to_partial_index_map[elem] if self.partial else elem
    #             self.total_similarity_with_subset[index] += self.ground_ground_kernel[elem, item]

    #     elif self.mode == "sparse":
    #         for elem in self.effective_ground_set:
    #             index = self.original_to_partial_index_map[elem] if self.partial else elem
    #             self.total_similarity_with_subset[index] += self.sparse_kernel.get_val(elem, item)

    def update_memoization(self, X: Set[int], item: int):
        if self.partial:
            effective_x = X.intersection(self.effective_ground_set)
        else:
            effective_x = X

        if item in effective_x or item not in self.effective_ground_set:
            return

        if self.mode == "dense":
            # Convert effective_ground_set to tensor
            effective_ground_set_tensor = torch.tensor(
                list(self.effective_ground_set), device=self.device, dtype=torch.long
            )
            indices = torch.tensor(
                [self.original_to_partial_index_map[elem] if self.partial else elem for elem in self.effective_ground_set],
                device=self.device
            )
            
            self.total_similarity_with_subset[indices] += self.ground_ground_kernel[effective_ground_set_tensor, item]

        elif self.mode == "sparse":
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                self.total_similarity_with_subset[index] += self.sparse_kernel.get_val(elem, item)


    def get_effective_ground_set(self) -> Set[int]:
        return self.effective_ground_set

    def clear_memoization(self):
        if self.mode == "dense" or self.mode == "sparse":
            self.total_similarity_with_subset = torch.zeros(self.num_effective_ground_set, device=self.device)
