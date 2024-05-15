# %%
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from tqdm import tqdm 
import time
from torch.utils.data import random_split, Dataset, DataLoader
#from torchvision.models.resnet import ResNet18_Weights
import pickle
import random
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import statistics
import matplotlib.pyplot as plt
import csv
from itertools import combinations, permutations
import pickle
import math
import numpy as np
from submodlib import FacilityLocationFunction, GraphCutFunction, DisparityMinFunction, DisparitySumFunction

seed = 42
torch.manual_seed(seed)

# %%
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:5" # change the available gpu number
else:
    device = "cpu"

# %% [markdown]
# ## Load Dataset

# %%
# Define data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# %%
# Load CIFAR10 datasets
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# %% [markdown]
# ## Parameters

# %%
print(len(train_dataset))

# %%
num_classes = 10
frac_size = 0.3
epochs = 40
R = 1
model_name = "LeNet"
optimizer_name = "adam"

# %% [markdown]
# ## Load Model

# %%
from models.LeNet_model import LeNet
from models.resent_models import get_resent101_model, get_resent18_model

# %%
from utils.milo.subset_sampler import RandomSubsetSampler
from utils.milo.subset_dataset import SubDataset

model = LeNet()

# %% [markdown]
# ## Experiment 

# %% [markdown]
# #### Optimizer

# %%
if optimizer_name=="SGD_ann":
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
elif optimizer_name=="adam":
    optimizer = torch.optim.Adam(model.parameters())

# %% [markdown]
# ## Preprocessing

# %%
import os
print(os.getcwd())

# %%
# func_list = ["facility-location", "disparity-min",  "disparity-sum", "graph-cut"]
func_list = ["fl", "dm",  "ds", "gc"]
lambdaVal = 0.5

# %%
embedding_path = "../milo-base/cifar10/dataframe.pkl"
with open(embedding_path, "rb") as f:
    df = pickle.load(f)

# %%
groups = df.groupby('Label')
dataframes = [group for _, group in groups]

# %%
objFL_list = []
objDM_list = []
objDS_list = []
objGC_list = []
index_list = []
for df in tqdm(dataframes):
    features = df["Features"].to_numpy()
    objFL = FacilityLocationFunction(n=features.shape[0], data=features, separate_rep=False, mode="dense", metric="cosine")
    objDM = DisparityMinFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
    objDS = DisparitySumFunction(n=features.shape[0], data=features, mode="dense", metric="cosine")
    objGC = GraphCutFunction(n=features.shape[0], data=features, mode="dense", metric="cosine", lambdaVal=lambdaVal)
    objFL_list.append(objFL)
    objDM_list.append(objDM)
    objDS_list.append(objDS)
    objGC_list.append(objGC)
    index_list.append(df.Index.tolist())

# %%
obj_list = {}
obj_list["fl"] = objFL_list
obj_list["dm"] = objDM_list
obj_list["ds"] = objDS_list
obj_list["gc"] = objGC_list

# %%
def class_SGE(k, obj_list, S, index_list, ϵ=1e-2, n=10, test=False):
    """Samples n subsets using stochastic-greedy.
    Args:
        k: The subset fraction.
        objL: submodular function for conditional gain calculation
        S: Prior Subset
        ϵ: The error tolerance.
        n: The number of classes.
    Returns:
        A list of n subsets.
    """
    
    # Set random subset size for the stochastic-greedy algorithm
    for i in range(n):
        D = len(index_list[i])
        sub_sz = int(D*k)
        s = int(D * math.log(1 / ϵ) / sub_sz)
        obj_list[i].evaluateWithMemoization(S[i])
        while len(S[i])<sub_sz:
            R = random.choices(list(set(range(D)) - S[i]), k=s)
            marginal_gains = list(map(lambda r: obj_list[i].marginalGainWithMemoization(S[i], r), R))
            max_index = np.argmax(marginal_gains)
            max_r = R[max_index]
            obj_list[i].updateMemoization(S[i], max_r)
            S[i].add(max_r)
            if test:
                print(R)
                print(marginal_gains)
                print(max_index, max_r)
                return S
        obj_list[i].clearMemoization()
    return S

# %% [markdown]
# ## Randon Submod Selection

# %%
if model_name=="LeNet":
    model = LeNet()
elif model_name=="resnet18":
    model = get_resent18_model()
elif  model_name=="resnet101":
    model = get_resent101_model()

model = model.to(device=device)
loss_fn = nn.CrossEntropyLoss()

if optimizer_name=="SGD_ann":
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
elif optimizer_name=="adam":
    optimizer = torch.optim.Adam(model.parameters())

# Train the model
model.train()
start_time = time.time()
accuracy_list = []

f1 = "ds"
f2 = "dm"
p=0.5

S1 = [set() for i in range(10)]
S2 = [set() for i in range(10)]

acc_dic = {}

# Train loop
for epoch in tqdm(range(epochs)):
    if epoch%R==0:
        remaining_elements = [e for e in func_list if e != f1 and e != f2]
        random_element = random.choice(remaining_elements)

        if random.random() < 0.5:
            f1 = random_element
        else:
            f2 = random_element
            
        S1 = class_SGE(k=frac_size, obj_list=obj_list[f1], S=S1, index_list=index_list)
        S2 = class_SGE(k=frac_size, obj_list=obj_list[f2], S=S2, index_list=index_list)
        
        bigS1 = []
        for i, s1 in enumerate(S1):
            bigS1.extend(index_list[i][j] for j in s1)
        
        bigS2 = []
        for i, s2 in enumerate(S2):
            bigS2.extend(index_list[i][j] for j in s2)

        train_data = list(set(bigS1) | set(bigS2))
        random.shuffle(train_data)
        if len(train_data) > int(frac_size*len(train_dataset)):
            train_data = train_data[:int(frac_size*len(train_dataset))]

        sub_dataset = SubDataset(indices=train_data, dataset=train_dataset)
        subset_train_dataloader = DataLoader(sub_dataset, batch_size=64, shuffle=True)

    for images, labels in subset_train_dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and update weights
        if optimizer_name=="SGD_ann":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model weights
            lr_scheduler.step()     
        elif optimizer_name=="adam":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if epoch%R==0:
        for i, s1 in enumerate(S1):
            s1 = list(s1)
            random.shuffle(s1)
            S1[i] = set(s1[:int(p*len(s1))])
        
        for i, s2 in enumerate(S2):
            s2 = list(s2)
            random.shuffle(s2)
            S2[i] = set(s2[:int(p*len(s2))])
    
    # Evaluate on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    accuracy_list.append(accuracy)
    # print(f"accuracy for epoch {epoch} with subset from {f1} and {f2}: {accuracy}")

time_taken = time.time() - start_time    
print("--- %s seconds ---" % (time_taken))

# Evaluate on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
acc_dic["online_is"] = accuracy_list
print(f"accuracy: {accuracy}")

# %%
print(accuracy_list)

# %% [markdown]
# ## 3-Step-Milo

# %%
func_orders = [
    ["fl", "gc", "fl", "dm", "dm", "ds"],
    ["fl", "gc", "dm", "ds", "fl", "dm"],
    ["fl", "dm", "fl", "gc", "dm", "ds"],
    ["fl", "dm", "dm", "ds", "fl", "gc"],
    ["dm", "ds", "fl", "dm", "fl", "gc"],
    ["dm", "ds", "fl", "gc", "fl", "dm"],
]

# %%
f1 = "ds"
f2 = "dm"
p=0

order_acc = {}

for order in func_orders:
    if model_name=="LeNet":
        model = LeNet()
    elif model_name=="resnet18":
        model = get_resent18_model()
    elif  model_name=="resnet101":
        model = get_resent101_model()

    model = model.to(device=device)
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name=="SGD_ann":
        optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif optimizer_name=="adam":
        optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    model.train()
    start_time = time.time()
    accuracy_list = []
    
    S1 = [set() for i in range(10)]
    S2 = [set() for i in range(10)]
    for epoch in tqdm(range(epochs)):
            
        if epoch==0:
            f1, f2 = order[0], order[1]
            update = True
        elif epoch==20:
            f1, f2 = order[2], order[3]
            update = True
        elif epoch==30:
            f1, f2 = order[4], order[5]
            update = True
            

        if update:
            update = False
            for i, s1 in enumerate(S1):
                s1 = list(s1)
                random.shuffle(s1)
                S1[i] = set(s1[:int(p*len(s1))])
            for i, s2 in enumerate(S2):
                s2 = list(s2)
                random.shuffle(s2)
                S2[i] = set(s2[:int(p*len(s2))])

            S1 = class_SGE(k=frac_size, obj_list=obj_list[f1], S=S1, index_list=index_list)
            S2 = class_SGE(k=frac_size, obj_list=obj_list[f2], S=S2, index_list=index_list)
            
            bigS1 = []
            for i, s1 in enumerate(S1):
                bigS1.extend(index_list[i][j] for j in s1)
            
            bigS2 = []
            for i, s2 in enumerate(S2):
                bigS2.extend(index_list[i][j] for j in s2)

            train_data = list(set(bigS1) | set(bigS2))
            random.shuffle(train_data)
            if len(train_data) > int(frac_size*len(train_dataset)):
                train_data = train_data[:int(frac_size*len(train_dataset))]

            sub_dataset = SubDataset(indices=train_data, dataset=train_dataset)
            subset_train_dataloader = DataLoader(sub_dataset, batch_size=64, shuffle=True)
        
        for images, labels in subset_train_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and update weights
            if optimizer_name=="SGD_ann":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # Update model weights
                lr_scheduler.step()     
            elif optimizer_name=="adam":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
        
        # Evaluate on test set
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        accuracy_list.append(accuracy)

    time_taken = time.time() - start_time    
    print("--- %s seconds ---" % (time_taken))

    # Evaluate on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    

    order_name = ""
    for n in order:
        order_name+=n+"_"
    
    order_name = order_name[:-1]
    order_acc[order_name] = accuracy
    print(f"order {order_name} accuracy: {accuracy}")
    print(accuracy_list)

    acc_dic[order_name] = accuracy_list

with open("./online_is.pkl", 'wb') as f:
    pickle.dump(acc_dic, f)