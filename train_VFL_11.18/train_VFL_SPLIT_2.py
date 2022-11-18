import random
import time
from datetime import datetime
from torch.utils.data.sampler import  WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from model.Linear_NN import *
from utils import *
from torch import nn
from sys import argv
import os
import argparse
import math
from torch.autograd import Variable
from Data.get_data import *
from sklearn.linear_model import LogisticRegression

# Define A random_seed
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
    parser = argparse.ArgumentParser(description='VFL1')
    parser.add_argument('--dataset', type=str, default='bank_marketing', help="dataset") # [bank_marketing, credit, census, cancer]
    parser.add_argument('--acti', type=str, default='leakyrelu_2', help="acti")  # [leakyrelu_2, leakyrelu_3, non]
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--noise_scale', default=0.01, type=float, help='batch_size')
    parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')  
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')  # [1e-4, ]
    return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


# Train_Server Side Program
def train_server(fx1, noise1, fx2, noise2, fx3, noise3, y, t, i, batch,correct):
    server_model.train()
    correct = correct
    size = size_train

    # Data 
    fx1 = Variable(fx1.to(device), requires_grad=True)
    fx2 = Variable(fx2.to(device), requires_grad=True)
    fx3 = Variable(fx3.to(device), requires_grad=True)
    y = y.to(device)
    

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model((fx1+noise1), (fx2+noise2), (fx3+noise3))


    # backward
    loss = criterion(fx_server, y)
    loss.backward()
    optimizer_server.step()


    # acc
    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch+1) * batch_size
    print(f"train-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%")

    return fx1.grad, fx2.grad, fx3.grad, correct


# Train_Client Side Program
def train_client(dataloader, client_model_1, client_model_2, client_model_3,  t):
    client_model_1.train()
    client_model_2.train()
    client_model_3.train()

    correct = 0
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:15]
          X3 = X[:, 15:]
      if dataset == 'credit':
        X1 = X[:, :7]
        X2 = X[:, 7:14]
        X3 = X[:, 14:]
      if dataset == 'census':
        X1 = X[:, :7]
        X2 = X[:, 7:23]
        X3 = X[:, 23:]
      if dataset == 'cancer':
        X1 = X[:, :3]
        X2 = X[:, 3:6]
        X3 = X[:, 6:]


      for i in range(1):

        # client1--train and update
        optimizer_client1.zero_grad()
        fx1 = client_model_1(X1, 1)
        noise1 = np.random.normal(0, math.sqrt(noise_scale), fx1.size())
        noise1 = torch.from_numpy(noise1).float().cuda()
        print('noise1', noise1)
        print('noise1.is_leaf', noise1.is_leaf)


        # client2--train and update
        optimizer_client2.zero_grad()
        fx2 = client_model_2(X2, 2)
        noise2 = np.random.normal(0,  math.sqrt(noise_scale), fx2.size())
        noise2 = torch.from_numpy(noise2).float().cuda()


        # client3--train and update
        optimizer_client3.zero_grad()
        fx3 = client_model_3(X3, 3)
        noise3 = np.random.normal(0,  math.sqrt(noise_scale), fx3.size())
        noise3 = torch.from_numpy(noise3).float().cuda()

        # Sending activations to server and receiving gradients from server
        g_fx1, g_fx2, g_fx3, correct = train_server(fx1,  noise1, fx2, noise2, fx3, noise3, y, t, i, batch, correct)   

  
        # backward prop
        (fx1 + noise1).backward(g_fx1) # 不允许张量对张量求导，只允许标量对张量求导：将所有张量元素加权求和转换为标量
        (fx2 + noise2).backward(g_fx2) # 因此分成两步：l = torch.sum(fx1*g_fx1) , 然后l 对 w求导；
        (fx3 + noise3).backward(g_fx3) 

        optimizer_client1.step()
        optimizer_client2.step()
        optimizer_client3.step()
        for param in client_model_1.parameters():
            #   print('client_model_1.p', param)
            print('client_model_1.p.grad', param.grad)
        

def test_server(client1_fx, client2_fx, client3_fx, y, batch, correct):
    server_model.eval()
    correct = correct
    size = size_test

    # Data
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    client3_fx = client3_fx.to(device)
    y = y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx, client3_fx)
    loss = criterion(fx_server, y)
    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()

    correct_train = correct / size
    loss, current = loss.item(), (batch+1) * len(client1_fx)
    print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%")

    return correct


# Test_Client Side Program
def test_client(dataloader, client_model_1, client_model_2,client_model_3, t):
    client_model_1.eval()
    client_model_2.eval()
    client_model_3.eval()
    correct = 0
    size = size_test
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:15]
          X3 = X[:, 15:]
        if dataset == 'credit':
          X1 = X[:, :7]
          X2 = X[:, 7:14]
          X3 = X[:, 14:]
        if dataset == 'census':
          X1 = X[:, :7]
          X2 = X[:, 7:23]
          X3 = X[:, 23:]
        if dataset == 'cancer':
          X1 = X[:, :3]
          X2 = X[:, 3:6]
          X3 = X[:, 6:]

        # client1--train and update
        optimizer_client1.zero_grad()
        fx1 = client_model_1(X1, 1)
        client1_fx = fx1.clone().detach().requires_grad_(True)
        # print('test-client1_fx', client1_fx)
        # print('test--norm(client1_fx)', client1_fx.norm())


        # client2--train and update
        optimizer_client2.zero_grad()
        fx2 = client_model_2(X2, 2)
        client2_fx = fx2.clone().detach().requires_grad_(True)
        # print('test-client2_fx', client2_fx)
        # print('test--norm(client2_fx)', client2_fx.norm())


        # client3--train and update
        optimizer_client3.zero_grad()
        fx3 = client_model_3(X3, 3)
        client3_fx = fx3.clone().detach().requires_grad_(True)


        # Sending activations to server and receiving gradients from server
        correct = test_server(client1_fx, client2_fx, client3_fx, y, batch, correct)

    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n")


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    batch_size = args.batch_size
    num_cutlayer = args.num_cutlayer
    noise_scale = args.noise_scale
    epochs = args.epochs
    lr = args.lr
    dataset=args.dataset
    acti = args.acti
    time_start_load_everything = time.time()

    # Define record path
    save_path = f'Results/{dataset}/n{noise_scale}/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    # Define data
    train_iter, test_iter, size_train, size_test = data(dataset, batch_size)


    # Define model
    if dataset == 'bank_marketing':
        if acti == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_em_bm(in_dim=5, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
          client_model_3 = Client_LeakyreluNet_2_em_bm(in_dim=5, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=3).to(device)
          server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*3, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

        if acti == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_em_bm(in_dim=5, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
          client_model_3 = Client_LeakyreluNet_3_em_bm(in_dim=5, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=3).to(device)
          server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*3, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

        if acti == 'linear':
          client_model_1 = Client_LinearNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LinearNet_2_em_bm(in_dim=5, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
          client_model_3 = Client_LinearNet_2_em_bm(in_dim=5, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=3).to(device)
          server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*3, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

        optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
        optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)
        optimizer_client3 = torch.optim.Adam(client_model_3.parameters(), lr=lr)
        optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)
  
    
    # Define criterion
    criterion = nn.CrossEntropyLoss()     

    # start training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_client(train_iter, client_model_1, client_model_2, client_model_3, t)
        test_client(test_iter, client_model_1, client_model_2, client_model_3, t)

    print("Done!")

    







