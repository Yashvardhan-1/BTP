
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


class DisparitySum_imp(SetFunction):

    def __init__(self, n, mode, sijs=None, data=None, metric="cosine", num_neighbors=None):
        super(DisparitySum_imp, self).__init__()

        self.n = n
        self.mode = mode
        self.metric = metric
        self.sijs = sijs
        self.data = data
        self.num_neighbors = num_neighbors
        self.effective_ground_set = None
        self.current_sum = 0.0

        if self.n <= 0:
            raise ValueError("Number of elements in the ground set must be positive")

        if self.mode not in ['dense', 'sparse']:
            raise ValueError("Mode must be either 'dense' or 'sparse'")

        if type(self.sijs) != type(None):
            # Handle user-provided similarity kernel
            if self.mode == "dense":
                self.sijs = torch.tensor(self.sijs)  # Convert to PyTorch tensor
            elif self.mode == "sparse":
                self.sijs = sparse.coo_matrix(self.sijs)  # Convert to PyTorch sparse tensor
                self.sijs = self.sijs.coalesce()  # Convert to COO format

            if type(self.data) != type(None):
                print("Warning: Provided similarity kernel found. Ignoring the provided data matrix.")

        else:
            if type(self.data) != type(None):
                # Handle case when data matrix is provided
                if self.mode == "dense":
                    self.sijs = create_kernel(torch.tensor(self.data), self.metric, mode="dense")
                elif self.mode == "sparse":
                    self.sijs = create_kernel(torch.tensor(self.data), self.metric, mode="sparse", num_neigh=self.num_neighbors)

        self.effective_ground_set = set(range(n))

    def evaluate(self, X: Set[int]) -> float:
        effective_X = torch.tensor(list(X))
        if len(effective_X) == 0:
            return 0.0
        if self.mode == 'dense':
            return get_sum_dense(effective_X, self.sijs)
        elif self.mode == 'sparse':
            return get_sum_sparse(effective_X, self.sijs)
        else:
            raise ValueError("Only 'dense' and 'sparse' modes are supported")

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        return self.current_sum

    def marginal_gain(self, X: Set[int], item: int) -> float:
        effective_X = torch.tensor(list(X))
        gain = 0.0

        if item in effective_X:
            return 0.0

        if item not in self.effective_ground_set:
            return 0.0

        if self.mode == 'dense':
            for elem in effective_X:
                gain += (1 - self.sijs[elem][item])
        elif self.mode == 'sparse':
            for elem in effective_X:
                gain += (1 - self.sijs[item, elem].item())  # Accessing element in sparse tensor
        else:
            raise ValueError("Only 'dense' and 'sparse' modes are supported")

        return gain

    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
        effective_X = torch.tensor(list(X))
        gain = 0.0

        if enable_checks and item in effective_X:
            return 0.0

        # There's no need for this condition as we don't use self.effective_ground_set anymore
        # if False and item not in self.effective_ground_set:
        #     return 0.0

        if self.mode == 'dense':
            for elem in effective_X:
                gain += (1 - self.sijs[elem][item])
        elif self.mode == 'sparse':
            for elem in effective_X:
                gain += (1 - self.sijs[item, elem].item())  # Accessing element in sparse tensor
        else:
            raise ValueError("Only 'dense' and 'sparse' mode supported")

        return gain

    def marginal_gain_with_memoization_vector(self, X: Set[int], R, enable_checks: bool = True):
        effective_X = torch.tensor(list(X))
        gain = torch.zeros_like(R, dtype=torch.float64)

        if self.mode == 'dense':
            for elem in effective_X:
                gain += (1 - self.sijs[elem][R])
        elif self.mode == 'sparse':
            for elem in effective_X:
                gain += (1 - self.sijs[R, elem].to(torch.float64))

        mask = torch.tensor([elem in effective_X for elem in R])  # Create mask for elements in effective_X
        gain[mask] = 0  # Set gains for elements in effective_X to 0

        return gain


    def update_memoization(self, X: Set[int], item: int) -> None:
        self.current_sum += self.marginal_gain(X, item)

    def clear_memoization(self) -> None:
        self.current_sum = 0.0

    def set_memoization(self, X: Set[int]) -> None:
        self.current_sum = self.evaluate(X)

