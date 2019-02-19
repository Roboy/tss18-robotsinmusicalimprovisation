import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal
import pypianoroll as ppr
from utils.utils import (getSlicedPianorollMatrixNp, cutOctaves, debinarizeMidi,
            addCuttedOctaves, pianorollMatrixToTempMidi, transposeNotesHigherLower)
from utils.NoteSmoother import NoteSmoother
from scipy.spatial.distance import hamming


def reconstruct(file_path, model, start_bar, end_bar,
                                    temperature=0.5, smooth_threshold=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model.train():
        model.eval()

    with torch.no_grad():
        sample_np = getSlicedPianorollMatrixNp(file_path)
        sample_np = transposeNotesHigherLower(sample_np)
        sample_np = cutOctaves(sample_np)
        sample_np = sample_np[start_bar:end_bar]
        sample = torch.from_numpy(sample_np).float()
        recon, embed, logvar = model(sample.view(-1,1,96,60).to(device))
        recon = torch.softmax(recon,dim=3)
        recon = recon.squeeze(1).cpu().numpy()
        # recon /= np.abs(np.max(recon))
        recon[recon < (1-temperature)] = 0

        sample_play = debinarizeMidi(sample_np, prediction=False)
        sample_play = addCuttedOctaves(sample_play)
        recon = debinarizeMidi(recon, prediction=True)
        recon = addCuttedOctaves(recon)

        recon_out = recon[0]
        sample_out = sample_play[0]
        if recon.shape[0] > 1:
            for i in range(recon.shape[0]-1):
                sample_out = np.concatenate((sample_out,sample_play[i+1]), axis=0)
                recon_out = np.concatenate((recon_out,recon[i+1]), axis=0)

    # plot with pypianoroll
    sample_plot = ppr.Track(sample_out)
    ppr.plot(sample_plot)
    recon_plot = ppr.Track(recon_out)
    ppr.plot(recon_plot)
    # smooth output
    smoother = NoteSmoother(recon_out, threshold=smooth_threshold)
    smoothed_seq = smoother.smooth()
    smoother_seq_plot = ppr.Track(smoothed_seq)
    ppr.plot(smoother_seq_plot)


    # print("INPUT")
    # pianorollMatrixToTempMidi(sample_out, show=True,showPlayer=True,autoplay=False)
    #
    # print("RECONSTRUCTION")
    # pianorollMatrixToTempMidi(recon_out, show=True,showPlayer=True,autoplay=False)


def interpolate(sample1_path, sample2_path, model, sample1_bar=0, sample2_bar=0,
                        temperature=0.5, smooth_threshold=0, play_loud=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model.train():
        model.eval()

    with torch.no_grad():
        sample1 = getSlicedPianorollMatrixNp(sample1_path)
        sample1 = transposeNotesHigherLower(sample1)
        sample1 = cutOctaves(sample1[sample1_bar])
        sample2 = getSlicedPianorollMatrixNp(sample2_path)
        sample2 = transposeNotesHigherLower(sample2)
        sample2 = cutOctaves(sample2[sample2_bar])

        #prepare for input
        sample1 = torch.from_numpy(sample1.reshape(1,1,96,60)).float().to(device)
        sample2 = torch.from_numpy(sample2.reshape(1,1,96,60)).float().to(device)

        # embed both sequences
        embed1, _ = model.encoder(sample1)
        embed2, _ = model.encoder(sample2)

        # for hamming distance
        recon1 = model.decoder(embed1)
        recon1 = torch.softmax(recon1,dim=3)
        recon1 = recon1.squeeze(0).squeeze(0).cpu().numpy()
        # recon1 /= np.max(np.abs(recon1))
        recon1[recon1 < (1-temperature)] = 0
        recon1 = debinarizeMidi(recon1, prediction=False)
        recon1 = addCuttedOctaves(recon1)
        recon1[recon1 > 0] = 1
        hamming1 = recon1.flatten()

        recon2 = model.decoder(embed2)
        recon2 = torch.softmax(recon2,dim=3)
        recon2 = recon2.squeeze(0).squeeze(0).cpu().numpy()
        # recon2 /= np.max(np.abs(recon2))
        recon2[recon2 < (1-temperature)] = 0
        recon2 = debinarizeMidi(recon2, prediction=False)
        recon2 = addCuttedOctaves(recon2)
        recon2[recon2 > 0] = 1
        hamming2 = recon2.flatten()

        hamming_dists1 = []
        hamming_dists2 = []

        for i, a in enumerate(range(0,11)):
            alpha = a/10.
            c = (1.-alpha) * embed1 + alpha * embed2

            # decode current interpolation
            recon = model.decoder(c)
            recon = torch.softmax(recon,dim=3)
            recon = recon.squeeze(0).squeeze(0).cpu().numpy()
            # recon /= np.max(np.abs(recon))
            recon[recon < (1-temperature)] = 0
            recon = debinarizeMidi(recon, prediction=False)
            recon = addCuttedOctaves(recon)
            if smooth_threshold:
                smoother = NoteSmoother(recon, threshold=smooth_threshold)
                recon = smoother.smooth()
            #for current hamming
            recon_hamm = recon.flatten()
            recon_hamm[recon_hamm > 0] = 1
            current_hamming1 = hamming(hamming1, recon_hamm)
            current_hamming2 = hamming(hamming2, recon_hamm)
            hamming_dists1.append(current_hamming1)
            hamming_dists2.append(current_hamming2)

            # plot piano roll
            if i==0:
                recon_plot = recon
            else:
                recon_plot = np.concatenate((recon_plot, recon), axis=0)

            print("alpha = {}".format(alpha))
            print("Hamming distance to sequence 1 is {}".format(current_hamming1))
            print("Hamming distance to sequence 2 is {}".format(current_hamming2))
            if play_loud:
                pianorollMatrixToTempMidi(recon, prediction=True, show=True,
                        showPlayer=False, autoplay=True)

        alphas = np.arange(0, 1.1, 0.1)
        fig, ax = plt.subplots()
        ax.plot(alphas, hamming_dists1)
        ax.plot(alphas, hamming_dists2)
        ax.grid()

        fig2, ax2 = plt.subplots()
        # recon_plot = ppr.Track(recon_plot)
        downbeats = [i*96 for i in range(11)]
        # recon_plot.plot(ax, downbeats=downbeats)
        ppr.plot_pianoroll(ax2, recon_plot, downbeats=downbeats)
        plt.show()

def sample(model, temperature=0.5, smooth_threshold=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model.train():
        model.eval()

    gaussian = Normal(torch.zeros(100), torch.ones(100))
    # print(gaussian.sample())

    with torch.no_grad():
        sample = gaussian.sample()
        recon = model.decoder(sample.unsqueeze(0).to(device))
        recon = torch.softmax(recon, dim=3)
        recon = recon.squeeze(0).squeeze(0).cpu().numpy()
        # recon /= np.max(np.abs(recon))
        recon[recon < (1-temperature)] = 0
        recon = debinarizeMidi(recon, prediction=False)
        recon = addCuttedOctaves(recon)
        if smooth_threshold:
            smoother = NoteSmoother(recon, threshold=smooth_threshold)
            recon = smoother.smooth()

        pianorollMatrixToTempMidi(recon, prediction=True, show=True,
                showPlayer=False, autoplay=True)
