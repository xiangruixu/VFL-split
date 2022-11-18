import random
import time
from datetime import datetime
from torch.utils.data.sampler import  WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from model.ML_NN import *
from Data.get_data import *
from utils import *
from torch import nn
from sys import argv
import os
import argparse



# Define random_seed
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_random_seed(1234)

# Define parameter
def parse_args():
    parser = argparse.ArgumentParser(description='bank_markting')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--acti', type=str, default='leakyrelu_2', help="acti")  # [leakyrelu_2, leakyrelu_3, non]
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--mode', default=1, type=str, help='mode_training')
    parser.add_argument('--noise_scale', default=0.01, type=float, help='batch_size')
    parser.add_argument('--dataset', type=str, default='bank_marketing', help="dataset") # [bank_marketing, credit, census, cancer]
    return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


# Define train
def train(dataloader, model):
    size = size_train
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X, noise_scale)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 10 == 0 and batch != 0:
            correct_train = correct/size_train
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100*correct_train):>0.1f}%")
        
            for param in model.parameters():
            #   print('model.p', param)
              print('model.p.grad', param.grad)
        

# Define test
def test(dataloader, model):
    size = size_test
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, noise_scale)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    time_start_load_everything = time.time()

    # Define parameter
    batch_size = args.batch_size
    epochs = args.epochs
    noise_scale = args.noise_scale
    dataset = args.dataset
    mode =args.mode

    
    # Define data
    train_iter, test_iter, size_train, size_test = data(dataset, batch_size)

    # Define model
    if mode == '1':
        model = LeakyreluNet_bm_1().to(device)

    if mode == '2':
        model = LeakyreluNet_bm_2().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # start training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_iter, model)
        test(test_iter, model)
    print("Done!")



