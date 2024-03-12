import threading
import queue
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset
import time
import pickle
import random
# from submodlib import FacilityLocationFunction
import math
import os
import argparse
import timeit
import cProfile
import pstats


# pip install fsspec==2023.9.2


# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features2(img, resnet):
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

def extract_features(img, resnet):
    """
    :param img: A CIFAR image
    :return: List of features
    """

    resnet.eval()  # Disables dropout and batch normalization layers
    
    # Apply the transformation and convert the image to a tensor
    img_tensor = transform(img).unsqueeze(0)
    output = resnet(img_tensor)

    last_layer = resnet.fc  # Access the final fully connected layer

    resnet.zero_grad()  # Clear any existing gradients
    output.backward(retain_graph=True)  # Retain graph for later access
    last_layer_gradients = last_layer.weight.grad
    
    return last_layer_gradients


# Create a threading.Event object
event = threading.Event()

def extract_features_resnet_threaded_cifar(cifar_dataset, num_threads=4):
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

def main(dataset):
    # Load the ResNet18 model
    resnet = torch.hub.load('pytorch/vision:v0.11.3', 'resnet18', pretrained=True)
    cifar_dataset = dataset["train"]
    df = get_features_dataset(cifar_dataset, resnet)
    return df

if __name__ == '__main__':
    # Load the ResNet18 model
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--dataset", help="what dataset to generate features", default="cifar10")
    parser.add_argument("--time", action="store_true", help="Description of the boolean argument")

    args = parser.parse_args()
    os.chdir("..")
    dataset = "cifar10"

    dataset = load_dataset(args.dataset)
    exit()
    resnet = torch.hub.load('pytorch/vision:v0.11.3', 'resnet18', pretrained=True)

    run_time = {}
    if args.time:
      cProfile.run('main(dataset)', filename="./feature_gen_time/single_threaded.prof")
      stats = pstats.Stats("./feature_gen_time/single_threaded.prof")
      print(f"Total Execution Time: {stats.total_tt} seconds")

      run_time["single_threaded"] = stats.total_tt

      with open(f"./feature_gen_time/run_time.pkl", "wb") as f:
        pickle.dump(run_time, f)
        f.close()

      cProfile.run('extract_features_resnet_threaded_cifar(dataset["train"])', filename="./feature_gen_time/4_threaded_.prof")
      stats = pstats.Stats("./feature_gen_time/single_threaded_.prof")
      print(f"Total Execution Time: {stats.total_tt} seconds")

      with open(f"./feature_gen_time/run_time.pkl", "wb") as f:
        pickle.dump(run_time, f)
        f.close()

    else:
      df = extract_features_resnet_threaded_cifar(dataset["train"])
      # sort the dataframe by index
      df = df.sort_values(by='Index')
      # Convert the 'features' column to a NumPy array
      df['Features'] = np.array(df['Features'])
      with open(f"{dataset}/dataframe-grad.pkl", "wb") as f:
          pickle.dump(df, f)