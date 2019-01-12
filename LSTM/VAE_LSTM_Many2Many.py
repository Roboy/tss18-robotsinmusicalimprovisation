#!/usr/bin/env python
# coding: utf-8

import matplotlib
# matplotlib.use('Agg')
import sys
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
from utils.VAE import VAE
from loadModel import loadModel, loadStateDict


def createDataset(dataset, seq_length=8):
    #cut to a multiple of seq_length
    X = [dataset[i:i+seq_length] for i in range(len(dataset)-seq_length)]
    return np.array(X)


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

    def forward(self, embed):
        h_t0 = self.hidden_init()
        c_t0 = self.hidden_init()

        lstm_input = torch.relu(self.i2h(embed))
        output, (h_t1, c_t1) = self.lstm(lstm_input, (h_t0, c_t0))
        output = torch.relu(self.h2o(output[:,:,:]))

        return embed, output


def train(epoch):
    model.train()
    train_loss = 0
    train_distance = 0
    criterion = nn.MSELoss()
    accuracy_criterion = nn.CosineSimilarity()
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

        # Normalize to mean 0 and std 1
        mean_batch = torch.mean(embedding)
        std_batch = torch.std(embedding)
        embedding_norm = (embedding - mean_batch) / std_batch

        g_truth = embedding[:,half_seq_length:,:]
        input_lstm = embedding[:,:half_seq_length,:]
        _ , output_lstm = model(input_lstm)

        loss = criterion(output_lstm, g_truth)
        loss.backward()

        train_loss += loss.item()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        with torch.no_grad():
            prediction = autoencoder_model.decoder(output_lstm.float().view(-1, model.input_size))
            prediction = prediction.view(-1, half_seq_length, 1, 96, 60)
            train_distance += np.linalg.norm(data.view(-1, model.seq_length, 1,
                96, 60)[:,half_seq_length:].cpu().numpy() - prediction.cpu().numpy())

        gradients = []
        weights = []
        for i, f in enumerate(model.parameters()):
            gradients.append(f.grad.cpu().data.numpy())
            weights.append(f.cpu().data.numpy())

        if(batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset)*model.seq_length,
                100. * batch_idx / len(train_loader),
                loss.item()/(half_seq_length)))

    # average train loss
    train_loss /= (batch_idx+1)*(half_seq_length)
    train_distance /= (batch_idx+1)
    print('====> Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss))

    return train_loss, train_distance, gradients, weights

def test(epoch):
    model.eval()
    test_loss = 0
    test_distance = 0
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
            # normalize
            mean_batch = torch.mean(embedding)
            std_batch = torch.std(embedding)
            embedding_norm = (embedding - mean_batch) / std_batch

            g_truth = embedding[:,half_seq_length:,:]
            input_lstm = embedding[:,:half_seq_length,:]
            _ , output_lstm = model(input_lstm)

            temp_loss = criterion(output_lstm, g_truth).item()
            test_loss += temp_loss

            prediction = autoencoder_model.decoder(output_lstm.float().view(-1, model.input_size))
            prediction = prediction.view(-1, half_seq_length, 1, 96, 60)
            test_distance += np.linalg.norm(data.view(-1, model.seq_length, 1,
                96, 60)[:,half_seq_length:].cpu().numpy() - prediction.cpu().numpy())


    # average test loss
    test_loss /= (i+1)*(half_seq_length)
    test_distance /= (i+1)

    print('====> Test set Loss: {:.4f}'.format(test_loss))
    print('')
    return test_loss, test_distance


if __name__ == '__main__':
    ############HYPERPARAMS#####################
    epochs = 100
    learning_rate = [1e-5]#[1e-3, 1e-4, 1e-5]
    batch_size = [50]
    seq_length = 8
    log_interval = 200 # Log/show loss per batch
    input_size = 100
    ############LSTM PARAMS#####################
    hidden_size = [64]#[128, 256, 512]
    lstm_layers = [2]#[2, 3]
    lr_decay = [0.5]#[1, 0.9, 0.5]
    lr_decay_step = 5
    datasets = ['../../nici_datasets/MAESTRO.npz']#,
                #'../../nici_datasets/YamahaPianoCompetition2002NoTranspose.npz']
                #'../../nici_datasets/MAESTRO.npz'],
                #'/media/EXTHD/niciDatanpzDatasets/WikifoniaTranspose12up12down.npz',
                #'/media/EXTHD/niciData/npzDatasets/YamahaPianoCompetition2002NoTranspose.npz']
    save_path = 'yamahapc2002_notp_TEST'
    ############################################
    ############################################
    i=1
    #manual grid search
    for dataset in datasets:
        for bs in batch_size:
            for lr in learning_rate:
                for lr_d in lr_decay:
                    for ll in lstm_layers:
                        for hs in hidden_size:
                            # try:
                            writer = SummaryWriter(log_dir=('plots/' + save_path))
                            writer.add_text("dataset", dataset, global_step=i)
                            writer.add_text("learning_rate", str(lr), i)
                            writer.add_text("learning_rate_decay", str(lr_d), i)
                            writer.add_text("learning_rate_decay_step", str(lr_decay_step), i)
                            writer.add_text("lstm_layers", str(ll), i)
                            writer.add_text("hidden_size", str(hs), i)
                            writer.add_text("batch_size", str(bs), i)

                            #load variational autoencoder
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            autoencoder_model = VAE()
                            path_to_model = '../VAE/checkpoints/yamahapc2002tpby60.pth'


                            # autoencoder_model = loadModel(autoencoder_model, path_to_model, dataParallelModel=False)
                            autoencoder_model = loadStateDict(autoencoder_model, path_to_model)
                            autoencoder_model = autoencoder_model.to(device)

                            # load dataset from npz
                            data = np.load(dataset)
                            train_dataset = data['train']#[0:10]
                            test_dataset = data['test']#[0:10]
                            data.close()
                            print("train set: {}".format(train_dataset.shape))
                            print("test set: {}".format(test_dataset.shape))
                            # print("valid set: {}".format(valid_dataset.shape))

                            train_dataset = createDataset(train_dataset, seq_length=seq_length)
                            test_dataset = createDataset(test_dataset, seq_length=seq_length)
                            # valid_dataset = createDataset(valid_dataset, seq_length=seq_length)

                            print('train_dataset {}'.format(train_dataset.shape))
                            print('test_dataset {}'.format(test_dataset.shape))

                            # train_dataset = train_dataset[0:1000]
                            train_dataset = torch.from_numpy(train_dataset)
                            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)

                            # test_dataset = test_dataset[0:100]
                            test_dataset = torch.from_numpy(test_dataset)
                            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

                            # valid_dataset = torch.from_numpy(valid_dataset)
                            # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

                            model = LSTM_Many2Many(batch_size=bs, seq_length=seq_length,
                                         input_size=input_size, hidden_size=hs,
                                         lstm_layers = ll).double().to(device)

                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            #optimizer = optim.RMSprop(model.parameters(),lr=learning_rate, momentum=0.9)

                            train_losses = []
                            test_losses = []
                            best_test_loss = np.inf

                            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_d)
                            for epoch in range(1, epochs + 1):
                                scheduler.step()
                                current_train_loss, current_train_distance, gradients, weights = train(epoch)
                                writer.add_scalar('loss/train_loss_epoch', current_train_loss, epoch)
                                for j, grad in enumerate(gradients):
                                    writer.add_histogram('gradients/grad{}'.format(j), grad, epoch)
                                for j, weight in enumerate(weights):
                                    writer.add_histogram('weights/weight{}'.format(j), weight, epoch)
                                writer.add_scalar('loss/train_distance', current_train_distance, epoch)
                                train_losses.append(current_train_loss)

                                current_test_loss, current_test_distance = test(epoch)
                                writer.add_scalar('loss/test_loss_epoch', current_test_loss, epoch)
                                writer.add_scalar('loss/test_distance', current_test_distance, epoch)
                                test_losses.append(current_test_loss)
                                if(current_test_loss < best_test_loss):
                                     best_test_loss = current_test_loss
                                     # torch.save(model,'/media/EXTHD/niciData/models/LSTM_GRIDSEARCH_' + str(hidden_size) + 'hidden_'+ str(epochs) + 'epochs_Many2Many_'+ str(i) +'.model')

                            writer.close()
                            i+=1
                            # except:
                            #     i+=1
                            #     writer.close()
                            #     continue
