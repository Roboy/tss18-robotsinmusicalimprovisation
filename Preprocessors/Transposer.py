#!/usr/bin/env python

# Transposer
# This script transposes your midi files up and down and saves it on your computer as new midi files in the range of C2 to B7.
# Decided to do this in a seperate file to speed up preprocessing of the autoencoder.

import music21
import glob
import pypianoroll as ppr
import os
import numpy as np
import argparse
np.set_printoptions(threshold=np.inf)
from utils.utilsPreprocessing import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set file path')
    parser.add_argument("--file_path", help='Path to your MIDI files', type=str)
    parser.add_argument("--tp_by", help='Set the value how much you want to transpose your MIDI files. Max: 60, Default: 30', type=int)
    parser.add_argument("--tp_step_size", help='Sets the value of how many semitones you want to jump between transpositions. Default: 2', type=int)


    args = parser.parse_args()

    if args.file_path:
        file_path = args.file_path + "/*.mid"
    else:
        print("Please set --file_path to folder where your MIDI files are!")
        sys.exit()

    if args.tp_by:
        tp_by = args.tp_by
    else:
        print("Chose default for transpositions, will transpose by 60 semitones.")
        tp_by = 60

    if args.tp_step_size:
        tp_step_size = args.tp_step_size
    else:
        print("Transpose step size will default to 2.")
        tp_step_size = 2



    files_list = glob.glob(file_path)

    for count, f in enumerate(files_list):
        a = ppr.Multitrack(f, beat_resolution=24)
        a = a.get_stacked_pianoroll()

        # CHECK WHICH NOTES ARE HIGHER THAN B6 / LOWER THAN C2
        # AND TRANSPOSE
        a = np.rollaxis(a,2)
        a = transposeNotesHigherLower(a)

        # cut octaves
        a = cutOctaves(a)
        
        for i in range(1, tp_by+1, tp_step_size):
            b = np.roll(a, i, axis = 2)
            b = addCuttedOctaves(b)
            newTrack1 = ppr.Multitrack()
            for track in b:
                tempTrack = ppr.Track(track)
                newTrack1.append_track(tempTrack)
            newTrack1.write((os.path.splitext(f)[0]+'TpBy{}'.format(i)))

        print("{}/{}".format(count+1,len(files_list)))