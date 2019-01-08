#!/usr/bin/env python
# coding: utf-8

import matplotlib
# matplotlib.use('Agg')
import sys
# sys.path.append('/home/nici/workspace/tss18-robotsinmusicalimprovisation/')
import numpy as np
import glob
import pypianoroll as ppr
import time
import music21
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from utils.utilsPreprocessing import *
from tensorboardX import SummaryWriter

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=50000)


############HYPERPARAMS#####################
epochs = 10
learning_rate = 1e-5
batch_size = 10
seq_length = 8
log_interval = 100 # Log/show loss per batch
input_size = 100
############LSTM PARAMS#####################
hidden_size = 256
lstm_layers = 2
############################################
############################################

writer = SummaryWriter()

#load variational autoencoder
from utils.VAE import VAE
from loadModel import loadModel, loadStateDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###VARIATIONAL CONV AUTOENCODER############
autoencoder_model = VAE()
path_to_model = '../pretrained/YamahaPC2002_VAE_Reconstruct_NoTW_20Epochs.model'
###########################################

autoencoder_model = loadModel(autoencoder_model, path_to_model, dataParallelModel=False)
#autoencoder_model = loadStateDict(autoencoder_model, path_to_model)
autoencoder_model = autoencoder_model.to(device)

#load dataset from npy
data = np.load('../../maestro-v1.0.0/maestro-v1.0.0_train.npy')
train_dataset = data[0:1000]


data = np.load('../../maestro-v1.0.0/maestro-v1.0.0_test.npy')
test_dataset = data[0:100]
# data.close()

data = np.load('../../maestro-v1.0.0/maestro-v1.0.0_valid.npy')
valid_dataset = data[0:100]
# data.close()

print("train set: {}".format(train_dataset.shape))
print("test set: {}".format(test_dataset.shape))
print("valid set: {}".format(valid_dataset.shape))

import pdb

def createDataset(dataset, seq_length=8):
    #cut to a multiple of seq_length
    X = [dataset[i:i+seq_length] for i in range(len(dataset)-seq_length)]
    return np.array(X)


train_dataset = createDataset(train_dataset, seq_length=seq_length)
test_dataset = createDataset(test_dataset, seq_length=seq_length)
valid_dataset = createDataset(valid_dataset, seq_length=seq_length)

print('train_dataset {}'.format(train_dataset.shape))
print('test_dataset {}'.format(test_dataset.shape))

# train_dataset = train_dataset[0:1000]
train_dataset = torch.from_numpy(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# test_dataset = test_dataset[0:100]
test_dataset = torch.from_numpy(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#test_dataset = test_dataset[0:100]
valid_dataset = torch.from_numpy(valid_dataset)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


class LSTM_Many2Many(nn.Module):
    def __init__(self, batch_size=7, lstm_layers=2, hidden_size=32, seq_length=7, input_size=100):
        super(LSTM_Many2Many, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.seq_length = seq_length
        self.input_size = input_size
        
        # LSTM
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers, batch_first=True, dropout=0.3)
        
        # LINEAR LAYERS
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size,self.input_size)     
    
    def reorder_batch(self, embedded_seqs):
        return torch.stack(torch.chunk(embedded_seqs, int(self.batch_size/self.seq_length)))
         
    def hidden_init(self):
        return torch.zeros(self.lstm_layers, self.batch_size, 
                           self.hidden_size, dtype=torch.double).to(device)

    def forward(self, embed, future = 0):
        h_t0 = self.hidden_init()
        c_t0 = self.hidden_init()
        
        lstm_input = self.i2h(embed)
        output, (h_t1, c_t1) = self.lstm(lstm_input, (h_t0, c_t0))
        output = self.h2o(output[:,:,:])
        
        return embed, output
    

model = LSTM_Many2Many(batch_size=batch_size, seq_length=seq_length, 
             input_size=input_size, hidden_size=hidden_size,
             lstm_layers = lstm_layers).double().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(model.parameters(),lr=learning_rate, momentum=0.9)


def train(epoch):
    model.train()
    train_loss = 0
    criterion = nn.MSELoss()
    half_seq_length = int(model.seq_length/2)
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        #float byte tensor
        data = data.float().to(device)
        data = data.view(-1,1,96,60)

        #embed data with autoencoder
        with torch.no_grad():
            mu, logvar = autoencoder_model.encoder(data)
    
        #prepare for input lstm
        mu = mu.view(model.batch_size, model.seq_length, 100)
        # writer.add_embedding(mu, metadata=) ?????????????????????
        
        embedding = mu.double()
        
        g_truth = embedding[:,half_seq_length:,:]
        input_lstm = embedding[:,:half_seq_length,:]
        _ , output_lstm = model(input_lstm, future = 0)

        loss = criterion(output_lstm, g_truth)
        loss.backward()
        train_loss += loss.item()
        
        #tensorboard
        writer.add_scalar('loss/train', loss.item(), epoch)
        
        optimizer.step()
        if(batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset)*model.seq_length,
                100. * batch_idx / len(train_loader),
                loss.item()/(half_seq_length)))
    
    # average train loss
    train_loss /= (batch_idx+1)*(half_seq_length)
    print('====> Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss))
    return train_loss
    
def test(epoch):
    model.eval()
    test_loss = 0
    criterion = nn.MSELoss()
    half_seq_length = int(model.seq_length/2)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            data = data.view(-1,1,96,60)
            mu, logvar = autoencoder_model.encoder(data)
            
            #prepare for input lstm
            mu = mu.view(model.batch_size, model.seq_length, 100)
            embedding = mu.double()

            g_truth = embedding[:,half_seq_length:,:]
            input_lstm = embedding[:,:half_seq_length,:]
            _ , output_lstm = model(input_lstm, future = 0)

            temp_loss = criterion(output_lstm, g_truth).item()
            test_loss += temp_loss
            writer.add_scalar('loss/test', temp_loss, epoch)


    # average test loss
    test_loss /= (i+1)*(half_seq_length)

    print('====> Test set Loss: {:.4f}'.format(test_loss))
    print('')
    return test_loss


# In[ ]:


# %matplotlib inline
import matplotlib.pyplot as plt

train_losses = []
test_losses = []
best_test_loss = np.inf
# plot_save_path = '../plots/LSTM_NEW_WikifoniaTP12_'+str(hidden_size)+'hidden_' + str(epochs) +'epoch_Many2Many.png'
plot_save_path = '../plots/LSTM_MAESTRO_'+str(hidden_size)+'hidden_' + str(epochs) +'epoch_Many2Many.png'
# plot_save_path = '../plots/LSTM_YamahaPCNoTP_'+str(hidden_size)+'hidden' + str(epochs) +'epoch_Many2Many.png'
# plot_save_path = '../plots/LSTM_YamahaPCTP60_'+str(hidden_size)+'hidden' + str(epochs) +'epoch_Many2Many.png'
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(1, epochs + 1):
    train_losses.append(train(epoch))
    
    current_test_loss = test(epoch)
    # test_losses.append(current_test_loss)
    # if(current_test_loss < best_test_loss):
    #     best_test_loss = current_test_loss
    #     torch.save(model,'/media/EXTHD/niciData/models/LSTM_WikifoniaTP12_' + str(hidden_size) + 'hidden_'+ str(epochs) + 'epochs_Many2Many_.model')
    
    if (epoch % 20 == 0):
        # for plots during training
        plt.plot(train_losses, color='red', label='Train Loss')
        plt.plot(test_losses, color='orange', label='Test Loss')
        plt.savefig(plot_save_path)
        # plt.show()   





plt.plot(train_losses, color='red', label='Train loss')
plt.plot(test_losses, color='orange', label='Test Loss')
plt.legend()
plt.savefig(plot_save_path)       

writer.export_scalars_to_json('./all_scalars.json')
writer.close()