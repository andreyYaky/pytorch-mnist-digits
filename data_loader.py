# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# loading the mnist dataset
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from PIL import Image
import torch
import numpy as np

# (trainX, trainy), (testX, testy) = load_classifier_data()
def load_classifier_data(device=None):
    # load dataset
    # trainX.shape = (60000, 28, 28) = 784
    # trainy.shape = (60000)
    # testX.shape = (10000, 28, 28) = 784
    # testy.shape = (10000)
    (trainX, trainy), (testX, testy) = mnist.load_data()

    # Add channels dim for Conv2D layer
    # (B, H, W) -> (B, Channels=1, H, W)
    # Also normalize from 0-255 to 0.0-1.0
    trainX = torch.tensor(trainX, device=device).unsqueeze(dim=1).float() / 255.0
    testX = torch.tensor(testX, device=device).unsqueeze(dim=1).float() / 255.0

    trainy = torch.tensor(trainy, device=device).long()
    testy = torch.tensor(testy, device=device).long()

    return (trainX, trainy), (testX, testy)

def load_classifier_image(image_path: str, device=None) -> torch.Tensor:

    input_image = Image.open(image_path)
    
    input_image_tensor = input_image.resize((28, 28))
    input_image_tensor = np.array(input_image_tensor)
    # (H, W, Channels=3)
    input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
    
    # (H, W, Channels=1)
    # Also normalize from 0-255 to 0.0-1.0
    input_image_tensor = input_image_tensor.mean(dim=2, keepdim=True) / 255.0

    # (B, H, W, Channels=1)
    input_image_tensor = input_image_tensor.unsqueeze(0)

    # (B, Channels=1, H, W)
    input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

    return input_image_tensor