import threading
import queue
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset
import time
import random
# from submodlib import FacilityLocationFunction
import math
import os

# Load the CIFAR-10 dataset
cifar10 = load_dataset("cifar10")
# cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img, resnet):
    """
    :param img: A CIFAR image
    :return: List of features
    """

    # Apply the transformation and convert the image to a tensor
    img_tensor = transform(img).unsqueeze(0)

    # Extract the features using the ResNet18 model
    with torch.no_grad():
        features = resnet(img_tensor)

    # Flatten the features and convert to a 1D numpy array
    features = features.squeeze().numpy()
    features = features.flatten()
    
    return features
    # return list(features)

# Create a threading.Event object
event = threading.Event()

def extract_features_resnet_threaded_cifar(cifar_dataset, num_threads=4 ):
    num_imgs = len(cifar_dataset)

    # Create a queue to store the image data
    img_queue = queue.Queue()

    # Create a list to store the extracted features
    features_list = []
    label_list = []
    index_list = []

    # Create a lock to protect the features list
    features_lock = threading.Lock()

    # Create a list of threads
    threads = []
    models = []

    # create multiple copies of the models for extracting features
    for i in range(num_threads):
        models.append(torch.hub.load('pytorch/vision:v0.11.3', 'resnet18', pretrained=True))

    # Start the threads
    for i in range(num_threads):
        thread = threading.Thread(target=extract_features_threaded_worker, args=(img_queue, index_list, features_list, label_list, features_lock, models[i], event))
        thread.start()
        threads.append(thread)

    # Enqueue all the image data
    for i, img in tqdm(enumerate(cifar_dataset)):
        img_queue.put((i, img["img"], img["label"]))

    # Signal to the threads that all of the images have been enqueued
    event.set()

    # Wait for all of the threads to finish
    for thread in threads:
        thread.join()

    # Create a DataFrame from the extracted features
    data = {'Index': index_list, 'Label': label_list, 'Features': features_list}
    df = pd.DataFrame(data)

    return df

def extract_features_threaded_worker(img_queue, index_list, features_list, label_list, features_lock, model, event):
    processed_images = 0
    while True:
        # Get an image path from the queue
        if img_queue.empty():
            # Wait for the main thread to signal that all of the images have been enqueued
            event.wait()

            # If the queue is still empty, break out of the loop
            if img_queue.empty():
                break

        index, img_path, label = img_queue.get()

        # Extract the features from the image
        img_features = extract_features(img_path, model)

        # Acquire the lock
        features_lock.acquire()

        # Add the extracted features to the list
        features_list.append(img_features)
        index_list.append(index)
        label_list.append(label)

        # Release the lock
        features_lock.release()
        
        # Increment the number of processed images
        processed_images+=1

        # If the thread has processed 1000 images, print the thread ID and the number of processed images
        if processed_images % 1000 == 0:
            print(f"Thread ID: {threading.current_thread().ident} | Processed images: {processed_images}")

        # If the queue is empty, break out of the loop
        if img_queue.empty():
            print(f"Thread ID: {threading.current_thread().ident} | Processed images: {processed_images}")
            break

def get_features_dataset(dataset, model):
    num_imgs = len(dataset)

    # Create a list to store the extracted features
    features_list = []
    label_list = []
    index_list = []

    # Enqueue all the image data
    for i, img in tqdm(enumerate(dataset)):
        features = extract_features(img["img"], model)
        features_list.append(features)
        label_list.append(img["label"])
        index_list.append(i)
        if i==5:
            break
        if i%100==0:
            print(time.strftime('%X'))

    # Create a DataFrame from the extracted features
    data = {'Index': index_list, 'Label': label_list, 'Features': features_list}
    df = pd.DataFrame(data)
    return df

def main():
    # Load the ResNet18 model
    resnet = torch.hub.load('pytorch/vision:v0.11.3', 'resnet18', pretrained=True)
    cifar_dataset = cifar10["train"]
    df = get_features_dataset(cifar_dataset, resnet)
    return df

if __name__ == '__main__':
    # df = main()
    # Load the ResNet18 model
    resnet = torch.hub.load('pytorch/vision:v0.11.3', 'resnet18', pretrained=True)

    df = extract_features_resnet_threaded_cifar(cifar10["train"])
    # sort the dataframe by index
    df = df.sort_values(by='Index')
    # Convert the 'features' column to a NumPy array
    df['Features'] = np.array(df['Features'])
    # Save the features DataFrame to a CSV file
    filename = "features_cifar_check.csv"
    df.to_csv(filename, index=False)
    
    np_array = np.array(df['Features'])
    flattened_array = np_array.tolist()
    # Create a DataFrame from the flattened array
    df = pd.DataFrame(flattened_array)
    # Save the DataFrame to a CSV file
    df.to_csv('np.csv', index=False, header=False)

    # MILO SGE
    # k=1000
    # '''
    #     I tried running objFL on the jupyter notebook. There the time to create the objFL is very large (kernel crashes)
    #     I have written the code for stochastic greedy following the MILO paper. 
    #     There A was not defined so I took it as S_i only. 
    #     stochastic_greedy() should give distinct subsets 
    # '''
    # objFL = FacilityLocationFunction(n=np_array.shape[0], data=np_array, separate_rep=False, mode="dense", metric="euclidean")
    # # S = stochastic_greedy(len(df), k, objFL=objFL)

    # output_directory = 'csv_files_yash/'
    # os.makedirs(output_directory, exist_ok=True)
    # for i, s in enumerate(S):
    #     s_list = list(s)
    #     df = pd.DataFrame({f'Set_{i}': s_list})
    #     filename = os.path.join(output_directory, f'set_{i}.csv')
    #     df.to_csv(filename, index=False, header=False)

    #     print(f'Set {i} saved to {filename}')

# def stochastic_greedy(D, k, objFL, ϵ=1e-2, n=10):
#     """Samples n subsets using stochastic-greedy.

#     Args:
#     D: number of training example
#     k: The subset size.
#     objFL: submodular function for conditional gain calculation
#     ϵ: The error tolerance.
#     n: The number of subsets to sample.

#     Returns:
#     A list of n subsets.
#     """
    
#     # Initialize empty subsets
#     S = [set() for _ in range(n)]
#     # Set random subset size for the stochastic-greedy algorithm
#     s = int(D * math.log(1 / ϵ) / k)
#     for i in range(n):
#         for j in range(k):
#             # Sample a random subset by sampling s elements from D \ Si
#             R = random.choices(list(set(range(D)) - S[i]), k=s)
#             # Use map to calculate marginal gains for all values in R with the custom set
#             marginal_gains = list(map(lambda r: objFL.marginalGain(S[i], r), R))
#             max_index = np.argmax(marginal_gains)
#             max_r = S[max_index]
#             S[i].add(max_r)

#     return S