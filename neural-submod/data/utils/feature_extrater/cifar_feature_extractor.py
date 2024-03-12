import threading
import queue
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
import time
import random
import math
import os

def extract_features(img, transform, model):
    """
        :param img: A CIFAR image
        :param transform: A CIFAR image
        :param model: A CIFAR image
        :return: List of features
    """

    # Apply the transformation and convert the image to a tensor
    img_tensor = transform(img).unsqueeze(0)

    # Extract the features using the ResNet18 model
    with torch.no_grad():
        features = model(img_tensor)

    # Flatten the features and convert to a 1D numpy array
    features = features.squeeze().numpy()
    features = features.flatten()
    
    return features


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

def extract_features_resnet_threaded_cifar(dataset, num_threads=4 ):
    # Create a threading.Event object
    event = threading.Event()
    
    num_imgs = len(dataset)

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
    for i, img in tqdm(enumerate(dataset)):
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
        if i%1000==999:
            print(time.strftime('%X'))

    # Create a DataFrame from the extracted features
    data = {'Index': index_list, 'Label': label_list, 'Features': features_list}
    df = pd.DataFrame(data)
    return df