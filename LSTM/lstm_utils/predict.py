"SCRIPT FOR CLEANER Predicter_VAE_LSTM_Many2One file"

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
from utils.utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(file_path, lstm_model, vae_model, temperature,
                    play_bar=0, bars=8, is_4bar_model=False):
    if(lstm_model.train()):
        lstm_model.eval()
    if(vae_model.train()):
        vae_model.eval()
    if is_4bar_model:
        bars = bars-3
    with torch.no_grad():
        sampleNp1 = getSlicedPianorollMatrixNp(file_path)
        sample = np.expand_dims(sampleNp1[0,:,36:-32],axis=0)
        for i, sampleNp in enumerate(sampleNp1[play_bar:play_bar+(bars-1)]):
            sampleNp = sampleNp[:,36:-32]
            sampleNp = np.expand_dims(sampleNp,axis=0)
            sample = np.concatenate((sample,sampleNp),axis=0)
        samplePlay = sample[0,:,:]
        for s in sample[1:-1]:
            samplePlay = np.concatenate((samplePlay,s),axis=0)
        samplePlay = addCuttedOctaves(samplePlay)

        #####PREPARE SAMPLE for input
        sample = torch.from_numpy(sample).float().to(device)
        sample = torch.unsqueeze(sample,1)

        #####MODEL##############
        embed, _ = vae_model.encoder(sample)
        embed = embed.unsqueeze(0).double()
        if is_4bar_model:
            _, lstm_out = lstm_model(embed[:,:4,:])
            lstm_out = lstm_out.float().squeeze(0)

        else:
            _, lstm_out = lstm_model(embed[:,:-1,:])
            lstm_out = lstm_out.float()
        pred = vae_model.decoder(lstm_out)
        ########################
        if is_4bar_model:
            pred = pred.squeeze(1)
            prediction = pred[0]
            for p in pred[1:]:
                prediction = torch.cat((prediction, p), dim=0)
            prediction = prediction.cpu().numpy()
        else:
            prediction = pred.squeeze(0).squeeze(0).cpu().numpy()

        #NORMALIZE PREDICTIONS
        #reconstruction /= np.abs(np.max(reconstruction))
        prediction /= np.abs(np.max(prediction))

        # temperature
        #reconstruction[reconstruction < 0.3] = 0
        prediction[prediction < (1-temperature)] = 0

        samplePlay = debinarizeMidi(samplePlay, prediction=False)
        samplePlay = addCuttedOctaves(samplePlay)
        #reconstruction = debinarizeMidi(reconstruction, prediction=True)
        #reconstruction = addCuttedOctaves(reconstruction)
        prediction = debinarizeMidi(prediction, prediction=True)
        prediction = addCuttedOctaves(prediction)

        # prediction_plot = ppr.Track(prediction)
        # prediction_plot.plot()

        print("INPUT")
        pianorollMatrixToTempMidi(samplePlay, show=True, showPlayer=True, autoplay=False)
        #print("RECONSTRUCTION")
        #pianorollMatrixToTempMidi(reconstruction, show=True,
        #                            showPlayer=True,autoplay=True, prediction=True)
        print("PREDICTION")
        pianorollMatrixToTempMidi(prediction, prediction=True,
                                  show=True,showPlayer=True,autoplay=True)
        print("\n\n")
