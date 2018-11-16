#!/usr/bin/env python
import numpy as np
import glob
import pypianoroll as ppr
import time
import music21
import os
import torch
import torch.utils.data
import argparse
from torch import nn, optim
from torch.nn import functional as F
from utils.utilsPreprocessing import *
from utils.LSTM import LSTM_Many2Many
from utils.VAE import VAE
from loadModel import loadModel
from LIVE.LiveInput_ClassCompliant import LiveParser

# argparser
parser = argparse.ArgumentParser(description='Hyperparameter and port selection')
parser.add_argument("--AE_model", help='Path to Autoencoder model. If you trained the AE on multiple GPUs use the --AE_model_dataParallel flag. Defaults to pretrained model', type=str)
parser.add_argument("--AE_is_dataParallel", help='Option to allow loading models trained with multiple GPUs. Default: False', action="store_true")
parser.add_argument("--LSTM_model", help='Path to LSTM model. Defaults to pretrained model', type=str)
parser.add_argument("--seq_length", help='Set the sequence length. Default: 8', type=int)
parser.add_argument("--hidden_size", help='Sets the hidden size of the LSTM model. This depends on the model you would like to load. Default: 128', type=int)
parser.add_argument("--input_size", help='If you trained an autoencoder with a different embedding size, you can change it here. Default: 100', type=int)

args = parser.parse_args()

#Autoencoder model
if args.AE_model:
    path_to_ae_model = args.AE_model
else:
    path_to_ae_model = "../../new_models_and_plots/YamahaPC2002_VAE_Reconstruct_NoTW_20Epochs.model"

#LSTM model
if args.LSTM_model:
    path_to_lstm_model = args.LSTM_model
else:
    path_to_lstm_model = "../../new_models_and_plots/LSTM_WikifoniaTP12_128hidden_180epochs_LRe-4_Many2Many.model"

#sequence length
if args.seq_length:
    seq_length = args.seq_length
else:
    seq_length = 4

#hidden_size
if args.hidden_size:
    hidden_size = args.hidden_size
else:
    hidden_size = 128

#input_size 
if args.input_size:
    input_size = args.input_size
else:
    input_size = 100

print(args.AE_is_dataParallel)
print(path_to_ae_model)
print(path_to_lstm_model)
print(seq_length)

def main():
    # check for gpu support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lstmModel = LSTM_Many2Many(batch_size=1, seq_length=seq_length*2, 
                     input_size=input_size, hidden_size=hidden_size)
    autoencoderModel = VAE()

    #load weights
    lstmModel = loadModel(lstmModel, path_to_lstm_model)
    autoencoderModel = loadModel(autoencoderModel, path_to_ae_model,
                                dataParallelModel=args.AE_is_dataParallel)
    # to device
    lstmModel = lstmModel.double().to(device)
    autoencoderModel = autoencoderModel.to(device)


    # Generate sample by feeding 7
    playSeq = 0
    lstmModel.batch_size = seq_length*2

    if(lstmModel.train()):
        lstmModel.eval()
    if(autoencoderModel.train()):
        autoencoderModel.eval()

    # get live input
    live_instrument = LiveParser(number_seq = seq_length)
    live_instrument.open_port(live_instrument.parse_notes)
    live_instrument.reset_clock()
    while (True):
        status_played_notes = live_instrument.clock()
        if status_played_notes:
            sequence = live_instrument.parse_to_matrix()
            break

    # send live recorded sequence through model and get improvisation
    with torch.no_grad():
        sample = np.array(np.split(sequence, seq_length))
        
        # prepare sample for input
        sample = cutOctaves(sample)
        sample = torch.from_numpy(sample).float().to(device)
        sample = torch.unsqueeze(sample,1)

        # model
        embed, _ = autoencoderModel.encoder(sample)
        embed = embed.unsqueeze(0).double()
        embed, lstmOut = lstmModel(embed, future=0)
        lstmOut = lstmOut.float()
        pred = autoencoderModel.decoder(lstmOut.squeeze(0))

        # reorder prediction
        pred = pred.squeeze(1)
        predict = pred[0]
        for p in pred[1:]:
            predict = torch.cat((predict, p), dim=0)
        predict = predict.cpu().numpy()

        # normalize predictions
        prediction = predict
        prediction /= np.abs(np.max(prediction))

        #check midi activations to include rests
        prediction[prediction < 0.3] = 0
        prediction = debinarizeMidi(prediction, prediction=True)
        prediction = addCuttedOctaves(prediction)

        print("Prediction")
        pianorollMatrixToTempMidi(prediction, prediction=True, 
                                  show=False,showPlayer=False,autoplay=True)        

if __name__ == '__main__':
    while(True):
        main()
