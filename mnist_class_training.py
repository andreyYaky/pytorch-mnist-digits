from data_loader import load_classifier_data
from classifier_models import CNN_Classifier
from classifier_models import UNET_Classifier

import torch
from torch import nn
import numpy as np
import torchmetrics

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False # Metal API for MacOS

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device {DEVICE}")

# load dataset
(trainX, trainy), (testX, testy) = load_classifier_data(device=DEVICE)

model = UNET_Classifier().to(DEVICE)

# Negative Log Likelihood
# like CrossEntropyLoss but assumes already LogSoftmax
loss_fn = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(DEVICE)

epochs = 100
batch_size = 100

model.train()

for n in range(epochs):
    # inline DataLoader
    ri = np.random.permutation(trainX.shape[0])[:batch_size]
    inputs, targets = trainX[ri], trainy[ri]
    predictions = model(inputs)

    loss = loss_fn(predictions, targets)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()
    
    # get accuracy
    acc = accuracy(predictions, targets)

    print(f"Step {n} loss : {loss}, accuracy: {acc * 100:.2f}%")

torch.save(model.state_dict(), "./data/state_dict_model.pt")