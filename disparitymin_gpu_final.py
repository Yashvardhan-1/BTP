import torch
import torch.nn as nn
from typing import List, Tuple, Set

class SetFunction(nn.Module):
    def __init__(self):
        super(SetFunction, self).__init__()

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
            return optimizer.maximize(self, budget, stopIfZeroGain, stopIfNegativeGain, verbose, show_progress, costs, cost_sensitive_greedy)
        else:
            print("Invalid Optimizer")
            return []

    def _get_optimizer(self, optimizer_name: str):
        if optimizer_name == "NaiveGreedy":
            return NaiveGreedyOptimizer()
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


def create_kernel_dense_sklearn(X, metric, X_rep=None, batch_size=1024):
    if X_rep is None:
        X_rep = X

    full_size = X.size(0)
    dense = torch.zeros(full_size, full_size, device=X.device)

    for start in range(0, full_size, batch_size):
        end = min(start + batch_size, full_size)
        if metric == "euclidean":
            dists = torch.cdist(X[start:end], X_rep)
            gamma = 1 / X.size(1)
            dense[start:end] = torch.exp(-dists * gamma)
        elif metric == "cosine":
            X_norm = X / X.norm(dim=1, keepdim=True)
            dense[start:end] = torch.mm(X_norm[start:end], X_norm.t())
        elif metric == "dot":
            dense[start:end] = torch.mm(X[start:end], X_rep.t())

    return dense

def create_sparse_kernel(X, metric, num_neigh,batch_size):
    if num_neigh > X.shape[0]:
        raise Exception("ERROR: num of neighbors can't be more than the number of datapoints")
    dense = create_kernel_dense_sklearn(X, metric,batch_size=batch_size )
    
    if num_neigh == -1:
        num_neigh = X.shape[0]  # default is the total number of datapoints

    distances = None
    if metric == 'euclidean':
        distances = torch.cdist(X, X, p=2)  # Euclidean distance
    elif metric == 'cosine':
        distances = 1 - torch.nn.functional.cosine_similarity(X.unsqueeze(1), X.unsqueeze(0), dim=2)  # Cosine similarity as distance

    distances.fill_diagonal_(float('inf'))
    _, ind = torch.topk(distances, k=num_neigh, largest=False)

    row, col = [], []
    for i, indices_row in enumerate(ind):
        for j in indices_row:
            row.append(i)
            col.append(j.item())

    mat = torch.zeros_like(distances)
    mat[row, col] = 1
    dense_ = dense * mat  # Only retain similarity of nearest neighbors
    sparse_coo = dense_.to_sparse()
    
    return sparse_coo

def create_kernel(X, metric, mode="dense", num_neigh=-1, X_rep=None, batch_size=1024):
    if X_rep is not None:
        assert X_rep.shape[1] == X.shape[1]

    if mode == "dense":
        return create_kernel_dense_sklearn(X, metric, X_rep, batch_size=batch_size)

    elif mode == "sparse":
        if X_rep is not None:
            raise Exception("Sparse mode is not supported for separate X_rep")
        return create_sparse_kernel(X, metric, num_neigh, batch_size=batch_size)

    else:
        raise Exception("ERROR: unsupported mode")

# class DisparityMin_imp(SetFunction):
#     def __init__(self, n, mode, sijs=None, data=None, metric="cosine", num_neighbors=None, batch_size = 0):
#         super(DisparityMin_imp, self).__init__()
#         self.n = n
#         self.mode = mode
#         self.metric = metric
#         self.sijs = sijs
#         self.data = data
#         self.num_neighbors = num_neighbors
#         self.effective_ground_set = None
#         self.currentMin = 0.0
#         self.batch_size = batch_size

#         if self.n <= 0:
#             raise Exception("ERROR: Number of elements in ground set must be positive")

#         if self.mode not in ['dense', 'sparse']:
#             raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")

#         if self.sijs is not None:
#             if isinstance(self.sijs, torch.sparse_coo_tensor):
#                 if num_neighbors is None:
#                     raise Exception("ERROR: num_neighbors must be given for sparse mode")
#                 self.num_neighbors = num_neighbors

#                 if self.sijs.shape != torch.Size([n, n]):
#                     raise Exception("ERROR: Inconsistent cardinality")

#                 if self.mode == "dense":
#                     self.sijs = self.sijs.to_dense()
#             elif isinstance(self.sijs, torch.Tensor):
#                 if self.sijs.shape != (n, n):
#                     raise Exception("ERROR: Inconsistent cardinality")

#                 if self.mode == "sparse":
#                     raise Exception("ERROR: Inconsistency between mode and kernel format")
#             else:
#                 raise Exception("Invalid kernel provided")
#         elif self.data is not None:
#             if isinstance(self.data, torch.Tensor):
#                 if self.data.shape[0] != n:
#                     raise Exception("ERROR: Inconsistent cardinality")

#                 if self.mode == "dense":
#                     self.sijs = create_kernel(self.data, metric, batch_size= self.batch_size)
#                 else:
#                     self.sijs = create_kernel(self.data, metric, mode, num_neighbors, batch_size = self.batch_size)
#             else:
#                 raise Exception("Invalid data type, must be torch.Tensor")

#         else:
#             raise Exception("ERROR: neither ground set data matrix nor similarity kernel provided")

#         self.effective_ground_set = set(range(n))

#         if self.mode == "dense":
#             self.cpp_disparity_min = self.sijs
#         else:
#             self.cpp_disparity_min = self.sijs

#     def evaluate(self, X: Set[int]) -> float:
#         if len(X) == 0:
#             return 0.0

#         subset_sijs = self.sijs[list(X), :][:, list(X)]
#         return torch.min(subset_sijs)

#     def marginal_gain(self, X: Set[int], item: int) -> float:
#         if item in X:
#             return 0.0

#         new_X = list(X) + [item]
#         subset_sijs = self.sijs[new_X, :][:, new_X]
#         return torch.min(subset_sijs) - self.evaluate(X)

#     def get_effective_ground_set(self) -> Set[int]:
#         return self.effective_ground_set

#     def clear_memoization(self) -> None:
#         self.currentMin = 0.0

#     def set_memoization(self, X: Set[int]) -> None:
#         if len(X) == 0:
#             self.currentMin = 0.0
#             return

#         subset_sijs = self.sijs[list(X), :][:, list(X)]
#         self.currentMin = torch.min(subset_sijs)

#     def update_memoization(self, X: Set[int], item: int) -> None:
#         if len(X) == 0:
#             self.currentMin = 0.0
#             return

#         new_X = list(X) + [item]
#         subset_sijs = self.sijs[new_X, :][:, new_X]
#         self.currentMin = torch.min(subset_sijs)

#     def evaluate_with_memoization(self, X: Set[int]) -> float:
#         return self.currentMin

#     def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
#         if enable_checks and item in X:
#             return 0.0

#         new_X = list(X) + [item]
#         subset_sijs = self.sijs[new_X, :][:, new_X]
#         return torch.min(subset_sijs) - self.evaluate_with_memoization(X)
class DisparityMin_imp(SetFunction):
    def __init__(self, n, mode, sijs=None, data=None, metric="cosine", num_neighbors=None, batch_size=None):
        super(DisparityMin_imp, self).__init__()
        self.n = n
        self.mode = mode
        self.metric = metric
        self.sijs = sijs
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size if batch_size is not None else 1024
        self.effective_ground_set = set(range(n))
        self.currentMin = 0.0

        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse']:
            raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")

        # Ensuring the data is on the right device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_sijs()

    def initialize_sijs(self):
        if self.data is not None:
            if not isinstance(self.data, torch.Tensor):
                raise ValueError("Data must be a PyTorch tensor")

            self.data = self.data.to(self.device)
            if self.mode == "dense":
                self.sijs = create_kernel_dense_sklearn(self.data, self.metric, batch_size=self.batch_size)
            elif self.mode == "sparse":
                self.sijs = create_sparse_kernel(self.data, self.metric, self.num_neighbors, batch_size=self.batch_size)
        else:
            raise ValueError("ERROR: Neither ground set data matrix nor similarity kernel provided")

    def evaluate(self, X: Set[int]) -> float:
        if len(X) == 0:
            return 0.0

        indices = torch.tensor(list(X), dtype=torch.long, device=self.device)
        subset_sijs = self.sijs[indices][:, indices]
        return torch.min(subset_sijs).item()

    def marginal_gain(self, X: Set[int], item: int) -> float:
        if item in X:
            return 0.0

        indices = torch.tensor(list(X) + [item], dtype=torch.long, device=self.device)
        subset_sijs = self.sijs[indices][:, indices]
        new_min = torch.min(subset_sijs).item()

        if len(X) == 0:
            current_min = 0.0
        else:
            indices = torch.tensor(list(X), dtype=torch.long, device=self.device)
            current_min = torch.min(self.sijs[indices][:, indices]).item()

        return new_min - current_min

    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
        if enable_checks and item in X:
            return 0.0

        new_X = list(X) + [item]
        subset_sijs = self.sijs[new_X, :][:, new_X]
        return torch.min(subset_sijs) - self.evaluate_with_memoization(X)

    def update_memoization(self, X: Set[int], item: int) -> None:
        indices = torch.tensor(list(X) + [item], dtype=torch.long, device=self.device)
        subset_sijs = self.sijs[indices][:, indices]
        self.currentMin = torch.min(subset_sijs).item()

    def clear_memoization(self) -> None:
        self.currentMin = 0.0

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        return self.currentMin

    def get_effective_ground_set(self) -> Set[int]:
        return self.effective_ground_set

    def set_memoization(self, X: Set[int]) -> None:
        if len(X) == 0:
            self.currentMin = 0.0
            return

        indices = torch.tensor(list(X), dtype=torch.long, device=self.device)
        subset_sijs = self.sijs[indices][:, indices]
        self.currentMin = torch.min(subset_sijs).item()