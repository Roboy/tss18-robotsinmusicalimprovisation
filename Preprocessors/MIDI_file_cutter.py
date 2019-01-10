# MIDI file cutter
# cuts midi files in 4 (default) sequences and writes to disk

import numpy as np
import glob
import time
import os
import pypianoroll as ppr
from utils.utilsPreprocessing import *
import argparse
import sys


def cut_single_track(track, file_length):
    too_long = track.shape[0] % file_length
    if(too_long != 0):
        track = track[:-too_long,:]
    sequences = np.array(np.vsplit(track, int(track.shape[0]/file_length)))
    return sequences


def write_to_disk(sequences, path):
    for i, seq in enumerate(sequences):
        if(np.any(seq)):
            tempTrack = ppr.Track(seq)
            newTrack = ppr.Multitrack()
            newTrack.append_track(tempTrack)
            newTrack.write(os.path.dirname(path)+'/sequences/'+os.path.splitext(os.path.basename(path))[0]+'_{}'.format(i))
        else:
            continue
  

def write_to_disk_npz(sequences, path):
    for i, seq in enumerate(sequences):
        if(seq.shape[0]!=96):
            print(seq.shape)
        if(np.any(seq)):
            #np.save(os.path.dirname(path)+'/sequences/'+os.path.splitext(os.path.basename(path))[0]+'_{}'.format(i), seq)
            tempTrack = ppr.Track(seq)
            newTrack = ppr.Multitrack()
            newTrack.append_track(tempTrack)
            newTrack.save(os.path.dirname(path)+'/sequences/'+os.path.splitext(os.path.basename(path))[0]+'_{}'.format(i))
        else:
            continue    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIDI file cutter. Cuts your MIDI files and puts them in a new directory')
    parser.add_argument("--file_path", help='Path to your MIDI files', type=str)
    parser.add_argument("--seq_length", default=96, help='Sequence length. Defaults to 96.', type=int)
    parser.add_argument("--bars", default=4, help='Number of bars you want to cut the MIDI file to.', type=int)
    parser.add_argument("--beat_res", default=24, help='Beat resolution of MIDI read', type=int)
    args = parser.parse_args()

    if not args.file_path:
        print("You did not use --file_path to link to your MIDI files")
        sys.exit()

    file_length = args.bars * args.seq_length
    os.mkdir(args.file_path + 'sequences')
    path_to_files = glob.glob(args.file_path + '*.mid')

    for i, path in enumerate(path_to_files):
        track = ppr.Multitrack(path, beat_resolution=args.beat_res)
        track = track.get_stacked_pianoroll()
        
        #single track midi file
        if(track.shape[2] == 1):
            track = np.squeeze(track,2)
            sequences = cut_single_track(track, file_length)
            write_to_disk(sequences, path)
            
        #multitrack midi file
        else:
            track = np.rollaxis(track,2,0)
            for track_number,instrument_track in enumerate(track):
                temp_path = os.path.splitext(path)[0]+'_'+str(track_number)+'_'+os.path.splitext(path)[1]
                sequences = cut_single_track(instrument_track, file_length)
                write_to_disk(sequences, temp_path)
        
        print('{}/{} tracks have been cut'.format(i+1,len(path_to_files)))


