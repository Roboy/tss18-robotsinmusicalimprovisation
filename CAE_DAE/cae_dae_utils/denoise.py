import matplotlib
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
from utils.NoteSmoother import NoteSmoother
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denoise(file_path, dae_model, play_bar=0,
                            play_it=False, smooth_threshold=0):
    if(dae_model.train()):
        dae_model.eval()

    with torch.no_grad():
        sample_np = getSlicedPianorollMatrixNp(file_path)
        sample_play = sample_np[play_bar]
        sample = sample_np[play_bar]
        sample = cutOctaves(sample)

        # prepare sample for input
        sample = torch.from_numpy(sample).float().to(device)
        sample = sample.unsqueeze(0).unsqueeze(0)
        # add noise
        noise = torch.bernoulli((torch.rand_like(sample))).to(device)
        noisy_sample = sample + noise

        # denoise sequence
        denoised_recon, mu = dae_model(noisy_sample)

        sample_play = debinarizeMidi(sample_play, prediction=False)
        sample_play = addCuttedOctaves(sample_play)

        noisy_sample = noisy_sample.squeeze(0).squeeze(0)
        noisy_sample = noisy_sample.cpu().numpy()
        noisy_sample = debinarizeMidi(noisy_sample, prediction=False)
        noisy_sample = addCuttedOctaves(noisy_sample)

        denoised_recon = denoised_recon.squeeze(0).squeeze(0)
        denoised_recon = denoised_recon.cpu().numpy()
        denoised_recon = debinarizeMidi(denoised_recon, prediction=True)
        denoised_recon = addCuttedOctaves(denoised_recon)

        # set all negative to 0 and all notes to velocity 127
        denoised_recon[denoised_recon < 0] = 0
        denoised_recon[denoised_recon > 0] = 127

        # plot with pypianoroll
        sample_plot = ppr.Track(sample_play)
        ppr.plot(sample_plot)

        noisy_sample_plot = ppr.Track(noisy_sample)
        ppr.plot(noisy_sample_plot)

        denoised_recon_plot = ppr.Track(denoised_recon)
        ppr.plot(denoised_recon_plot)

        # smooth output
        smoother = NoteSmoother(denoised_recon, threshold=smooth_threshold)
        smoothed_seq = smoother.smooth()
        smoother_seq_plot = ppr.Track(smoothed_seq)
        ppr.plot(smoother_seq_plot)


        if play_it:
            print("INPUT")
            pianorollMatrixToTempMidi(sample_play,
                                    show=True, showPlayer=True, autoplay=True)
            print("DENOISED RECONSTRUCTION")
            pianorollMatrixToTempMidi(denoised_recon, prediction=True,
                                  show=True,showPlayer=True,autoplay=True)
            print("\n\n")
