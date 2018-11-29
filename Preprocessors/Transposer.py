#!/usr/bin/env python

# Transposer
# This script transposes your midi files up and down and saves it on your computer as new midi files in the range of C2 to B7.
# Decided to do this in a seperate file to speed up preprocessing of the autoencoder.

import music21
import glob
import pypianoroll as ppr
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
from utils.utilsPreprocessing import *


filesList = glob.glob('/Volumes/EXT/DATASETS/WikifoniaDatabase/train/*.mid')

for count, f in enumerate(filesList):
    #print(f)
    a = ppr.Multitrack(f, beat_resolution=24)
    a = a.get_stacked_pianoroll()
    #print(a.shape)
    """CHECK WHICH NOTES ARE HIGHER THAN B6 / LOWER THAN C2
    #AND TRANSPOSE"""
    a = np.rollaxis(a,2)
    a = transposeNotesHigherLower(a)

    """CUT OCTAVES"""
    #test1 = a
    a = cutOctaves(a)
    #test2 = addCuttedOctaves(a)
    
    #print(np.array_equal(test1,test2))
    
    for i in range(1,60):
        #print(a.shape)
        b = np.roll(a, i, axis = 2)
        b = addCuttedOctaves(b)
        #c = np.roll(a, -i, axis = 2)
        #c = addCuttedOctaves(c)
        #print(b.shape)
        newTrack1 = ppr.Multitrack()
        #newTrack2 = ppr.Multitrack()
        for track in b:
            tempTrack = ppr.Track(track)
            newTrack1.append_track(tempTrack)
        #for track in c:
        #    tempTrack = ppr.Track(track)
        #    newTrack2.append_track(tempTrack)
        newTrack1.write((os.path.splitext(f)[0]+'TpBy{}'.format(i)))
        #newTrack2.write((os.path.splitext(f)[0]+'TpDown{}'.format(i)))

    print("{}/{}".format(count,len(filesList)))


# 
