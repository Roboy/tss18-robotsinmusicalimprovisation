import os
import numpy as np
import torch
import torch.utils.data as data
import pypianoroll as ppr
import glob
from utils.utils import transposeNotesHigherLower, cutOctaves


class createDatasetAE(data.Dataset):
    def __init__(self, file_path, beat_res=24, seq_length=96, binarize=True,
                                            bars=1, verbose=False):
        self.all_files = glob.glob(file_path + '*.mid')
        self.beat_res = beat_res
        self.bars = bars
        self.binarize = binarize
        self.seq_length = seq_length
        self.verbose = verbose

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        try:
            # load song from midi files and parse to numpy
            track = ppr.Multitrack(self.all_files[idx], beat_resolution=self.beat_res)
            track = track.get_stacked_pianoroll()

            # if: 1 track midifile
            # else: quick fix for multitrack, melody in almost every song on midi[0]
            if track.shape[2]==1:
                track = np.squeeze(track,2)
            else:
                track = track[:,:,0]

            # if length differs from seq_length, cut it to seq_length
            if track.shape[0] > self.seq_length:
                track = track[:4*self.beat_res*self.bars]
            elif track.shape[0] < self.seq_length:
                pad_with = self.seq_length - track.shape[0]
                temp = np.zeros((pad_with, 128), dtype=np.uint8)
                track = np.concatenate((track, temp))

            # binarize
            if self.binarize:
                track[track > 0] = 1

            # transpose notes out of range of the 5 chosen octaves
            sequence = transposeNotesHigherLower(track)
            # cut octaves to get input shape [96,60]
            sequence = cutOctaves(sequence)
            # unsqueeze first dimension for input
            sequence = np.expand_dims(sequence, axis=0)
        except:
            if self.verbose:
                print("MIDI file warning. Skipped a MIDI file because was not working properly.")
            sequence = np.zeros((1, self.seq_length, 60), dtype=np.uint8)
        return torch.from_numpy(sequence)


def loadDatasets(file_path, validation_path, batch_size, beat_resolution, bars=1,
                    seq_length=96, binarize=True, shuffle=True, drop_last=True):
    if os.path.isdir(file_path + 'train/') and os.path.isdir(file_path + 'test/'):
        print("train/ and test/ folder exist!")
        train_dataset = createDatasetAE(file_path + 'train/',
                                  beat_res = beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=binarize)

        test_dataset = createDatasetAE(file_path + 'test/',
                                  beat_res=beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=binarize)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=shuffle, drop_last=drop_last)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=shuffle, drop_last=drop_last)

    else:
        print("Only one folder with all files exist, using {}".format(file_path))
        dataset = createDatasetAE(file_path,
                                  beat_res=beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=binarize)
        train_size = int(np.floor(0.95 * len(dataset)))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                        [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                    batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    # validation set
    if validation_path:
        print("Path to validation set was set!")
        valid_dataset = createDatasetAE(validation_path,
                                  beat_res = beat_resolution,
                                  bars=bars,
                                  seq_length=seq_length,
                                  binarize=binarize)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        print("Please use --valiation_path to set path to validation set.")
        sys.exit()

    return train_loader, test_loader, valid_loader
