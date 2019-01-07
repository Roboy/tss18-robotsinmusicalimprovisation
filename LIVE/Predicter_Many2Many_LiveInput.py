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
from mido import MidiFile, Message

# argparser
parser = argparse.ArgumentParser(description='Hyperparameter and port selection')
parser.add_argument("--port", help='MIDI controller input port. If you want to set it from bash. You can also set it later. Default = None.', type=str)
parser.add_argument("--AE_model", help='Path to Autoencoder model. If you trained the AE on multiple GPUs use the --AE_model_dataParallel flag. Defaults to pretrained model.', type=str)
parser.add_argument("--AE_is_dataParallel", help='Option to allow loading models trained with multiple GPUs. Default: False', action="store_true")
parser.add_argument("--LSTM_model", help='Path to LSTM model. Defaults to pretrained model', type=str)
parser.add_argument("--temperature", help='Temperature of improvisation. Default: 0.7', type=float)
parser.add_argument("--seq_length", help='Set the sequence length. Default: 4', type=int)
parser.add_argument("--hidden_size", help='Sets the hidden size of the LSTM model. This depends on the model you would like to load. Default: 128', type=int)
parser.add_argument("--input_size", help='If you trained an autoencoder with a different embedding size, you can change it here. Default: 100', type=int)

args = parser.parse_args()

# port
if args.port:
    port = args.port
else:
    port = None
# Autoencoder model
if args.AE_model:
    path_to_ae_model = args.AE_model
else:
    path_to_ae_model = "../pretrained/YamahaPC2002_VAE_Reconstruct_NoTW_20Epochs.model"

# LSTM model
if args.LSTM_model:
    path_to_lstm_model = args.LSTM_model
else:
    path_to_lstm_model = "../pretrained/LSTM_WikifoniaTP12_128hidden_180epochs_LRe-4_Many2Many.model"
# temperature
if args.temperature:
    temperature = args.temperature
else:
    temperature = 0.7

# sequence length
if args.seq_length:
    seq_length = args.seq_length
else:
    seq_length = 4

# hidden_size
if args.hidden_size:
    hidden_size = args.hidden_size
else:
    hidden_size = 128

# input_size 
if args.input_size:
    input_size = args.input_size
else:
    input_size = 100

# print(args.AE_is_dataParallel)
# print(path_to_ae_model)
# print(path_to_lstm_model)
# print(seq_length)

def main(live_instrument, lstm_model, autoencoder_model):
    # check for gpu support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # to device
    lstm_model = lstm_model.double().to(device)
    autoencoder_model = autoencoder_model.to(device)


    # Generate sample by feeding 7
    playSeq = 0
    lstm_model.batch_size = seq_length*2

    if(lstm_model.train()):
        lstm_model.eval()
    if(autoencoder_model.train()):
        autoencoder_model.eval()

    # reset live input clock
    print("\nUser input\n")
    live_instrument.reset_sequence()
    live_instrument.reset_clock()
    while True:
        status_played_notes = live_instrument.clock()
        if status_played_notes:
            sequence = live_instrument.parse_to_matrix()
            live_instrument.reset_sequence()
            break

    # send live recorded sequence through model and get improvisation
    with torch.no_grad():
        sample = np.array(np.split(sequence, seq_length))
        
        # prepare sample for input
        sample = cutOctaves(sample)
        sample = torch.from_numpy(sample).float().to(device)
        sample = torch.unsqueeze(sample,1)

        # model
        embed, _ = autoencoder_model.encoder(sample)
        embed = embed.unsqueeze(0).double()
        embed, lstmOut = lstm_model(embed, future=0)
        lstmOut = lstmOut.float()
        pred = autoencoder_model.decoder(lstmOut.squeeze(0))

        # reorder prediction
        pred = pred.squeeze(1)
        predict = pred[0]
        for p in pred[1:]:
            predict = torch.cat((predict, p), dim=0)
        predict = predict.cpu().numpy()

        # normalize predictions
        prediction = predict
        prediction /= np.abs(np.max(prediction))

        # check midi activations to include rests
        prediction[prediction < (1-temperature)] = 0
        prediction = debinarizeMidi(prediction, prediction=True)
        prediction = addCuttedOctaves(prediction)

        print("\nPrediction\n")
        # play predicted sequence note by note
        live_instrument.reset_clock()
        play_tick = -1
        old_midi_on = np.zeros(1)
        played_notes = []
        while True:
            done = live_instrument.computer_clock()
            if live_instrument.current_tick > play_tick:
                play_tick = live_instrument.current_tick
                midi_on = np.argwhere(prediction[play_tick] > 0)
                if midi_on.any():
                    for note in midi_on[0]:
                        if note not in old_midi_on:
                            live_instrument.out_port.send(Message('note_on', note=note, velocity=127))
                            played_notes.append(note)
                else:
                    for note in played_notes:
                        live_instrument.out_port.send(Message('note_off', note=note))#, velocity=100))
                        played_notes.pop(0)

                if old_midi_on.any():
                    for note in old_midi_on[0]:
                        if note not in midi_on:
                            live_instrument.out_port.send(Message('note_off', note=note))
                
                old_midi_on = midi_on

            if done:
                break

    live_instrument.reset_sequence()

if __name__ == '__main__':
    # get live input
    live_instrument = LiveParser(port = port, number_seq = seq_length)
    live_instrument.open_inport(live_instrument.parse_notes)
    live_instrument.open_outport()

    lstm_model = LSTM_Many2Many(batch_size=1, seq_length=seq_length*2, 
                     input_size=input_size, hidden_size=hidden_size)
    autoencoder_model = VAE()

    #load weights
    lstm_model = loadModel(lstm_model, path_to_lstm_model)
    autoencoder_model = loadModel(autoencoder_model, path_to_ae_model,
                                dataParallelModel=args.AE_is_dataParallel)

    while(True):
        main(live_instrument, lstm_model, autoencoder_model)