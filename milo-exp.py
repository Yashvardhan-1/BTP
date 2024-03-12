from torchvision.models import resnet18
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from tqdm import tqdm 
import time
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.models.resnet import ResNet18_Weights
import pickle
import random
import statistics


seed = 42
torch.manual_seed(seed)

model_name = "resnet101"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:5" # change the available gpu number
else:
    device = "cpu"

def get_resent18_model(num_classes=10):
    model = torchvision.models.resnet18(weights=None)  # Use 'weights' for pretrained models
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def get_resent101_model(num_classes=10):
    model = torchvision.models.resnet101(weights=None)  # Use 'weights' for pretrained models
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

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

# Load CIFAR10 datasets
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

class LeNet(nn.Module):    
    def __init__(self, out_classes=10):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400,120),  #in_features = 16 x5x5 
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, out_classes),
            # nn.Softmax(dim=1)
        )
        
    def forward(self,x): 
        a1=self.feature_extractor(x)
        # print(a1.shape)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        return a2
    
class SubDataset(Dataset):
    def __init__(self, indices, dataset):
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        data_point = self.dataset[index]
        return data_point
    
store = {}

frac_list = [0.05, 0.10, 0.15, 0.3, 0.5]
func_list = ["facility-location", "graph-cut", "disparity-min", "disparity-sum"]
R = 10
device = "cuda:2"
epochs = 200
metric = "euclidean"
    
for func in func_list:
    store[func] = {}
    for subset_fraction in frac_list:
        num_classes = 10
        class_data = []
        
        try:
            for i in range(num_classes):
                with open(f"milo-base/cifar10/SGE-{metric}/{func}/class-data-{subset_fraction}/class_{i}.pkl", "rb") as f:
                    S = pickle.load(f)
                    class_data.append(S)
        except:
            print("sorry")
            continue

        num_sets = len(class_data[0])
        data = []

        for i in range(num_sets):
            S = []
            for j in range(num_classes):
                S.extend(class_data[j][i])
            data.append(S)
        
        print(len(data[0]))

        torch.manual_seed(42)

        if model_name=="LeNeT":
            model = LeNet()
        elif model_name=="resent18":
            model = get_resent18_model()
        elif  model_name=="resnet101":
            model = get_resent101_model()
        
        model = model.to(device)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # Train the model
        model.train()
        start_time = time.time()
        for epoch in tqdm(range(epochs)):
            
            # Train loop
            if epoch%R==0:
                sub_dataset = SubDataset(indices=data[epoch//R], dataset=train_dataset)
                subset_train_dataloader = DataLoader(sub_dataset, batch_size=64, shuffle=True)
                
            for images, labels in subset_train_dataloader:

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                # Backward pass and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        print(f"Accuracy {func} {subset_fraction}: {accuracy:.4f}")

        store[func][subset_fraction] = accuracy
        with open(f"store-{model_name}.pkl", "wb") as f:
            pickle.dump(store, f)
            f.close()
        