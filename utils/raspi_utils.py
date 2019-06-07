import numpy as np
import torch
import torch.utils.data as data
import music21
import glob
import time


def transposeNotesHigherLower(a):
    # This function transposes all notes higher than B6 and lower than C2
    # into 5 octave range from C2 to B6
    if(a.ndim>2):
        for i,track in enumerate(a):
            octCheck = np.argwhere(track>0)
            for j in octCheck:
                #print(j)
                if(j[1]>=36 and j[1]<96):
                    continue
                elif(j[1]<36 and j[1]>=24):
                    #print('There is a note below 36')
                    a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],12)
                elif(j[1]<24 and j[1]>=12):
                    #print('There is a note below 24')
                    a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],24)
                elif(j[1]<12):
                    #print('There is a note below 12')
                    a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],36)
                elif(j[1]>=96 and j[1]<108):
                    #print('There is a note above 96')
                    a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],-12)
                elif(j[1]>=108 and j[1]<120):
                    #print('There is a note above 108')
                    a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],-24)
                elif(j[1]>=120):
                    #print('There is a note above 120')
                    a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],-36)
    else:
        octCheck = np.argwhere(a>0)
        for j in octCheck:
            if(j[1]>=36 and j[1]<96):
                continue
            elif(j[1]<36 and j[1]>=24):
                #print('There is a note below 36')
                a[j[0]:j[0]+1,:] = np.roll(a[j[0]:j[0]+1,:],12)
            elif(j[1]<24 and j[1]>=12):
                #print('There is a note below 24')
                a[j[0]:j[0]+1,:] = np.roll(a[j[0]:j[0]+1,:],24)
            elif(j[1]<12):
                #print('There is a note below 12')
                a[j[0]:j[0]+1,:] = np.roll(a[j[0]:j[0]+1,:],36)
            elif(j[1]>=96 and j[1]<108):
                #print('There is a note above 96')
                a[j[0]:j[0]+1,:] = np.roll(a[j[0]:j[0]+1,:],-12)
            elif(j[1]>=108 and j[1]<120):
                #print('There is a note above 108')
                a[j[0]:j[0]+1,:] = np.roll(a[j[0]:j[0]+1,:],-24)
            elif(j[1]>=120):
                #print('There is a note above 120')
                a[j[0]:j[0]+1,:] = np.roll(a[j[0]:j[0]+1,:],-36)

    return a


def cutOctaves(tensor):
    if (tensor.ndim==3):
        tensor = tensor[:,:,36:-32]
    elif(tensor.ndim==2):
        tensor = tensor[:,36:-32]
    else:
        print("WARNING cutOctaves function")
        tensor = tensor[:,:,:,36:-32]
    return tensor


def addCuttedOctaves(matrix):
    if(matrix.shape[1]!=128):
        if(matrix.ndim==3):
            matrix = np.pad(matrix,[[0,0],[0,0],[36,32]],'constant')
        else:
            matrix = np.pad(matrix,[[0,0],[36,32]],'constant')
    return matrix


def pianorollMatrixToTempMidi(matrix, path='../utils/midi_files/temp.mid', prediction=True,
    show=False, showPlayer=False, autoplay=False):
    # matrix must be of LENGTHxPITCH dimension here: (96 or more,128)
    if(prediction):
        matrix[-3:,:] = 0

    tempTrack = ppr.Track(matrix)
    newTrack = ppr.Multitrack()
    newTrack.append_track(tempTrack)
    newTrack.write(path)

    score = music21.converter.parse(path)
    if(show):
        score.show()
    if(showPlayer):
        score.show('midi')
    if(autoplay):
        music21.midi.realtime.StreamPlayer(score).play()


def debinarizeMidi(a, prediction=True,velocity=127):
    if(prediction):
        # MONOPHONIC
        #rowMax = a.max(axis=1).reshape(-1, 1)
        #a[:] = np.where((a == rowMax) & (a > 0), velocity, 0)
        # POLYPHONIC
        min_vel = 80
        a[a > 0] = min_vel + (velocity-min_vel)*a[a > 0]

        # import pdb; pdb.set_trace()
    elif(prediction==False):
        a[:] = np.where(a > 0,velocity,0)
    return a


def noteThreshold(a, threshold=0.5, velocity=127):
    a[:] = np.where(a > threshold, velocity, 0)
    return a


def transposeTracks(midiFiles):
    #TRANPOSE OCTAVE BELOW AND ABOVE OF ORIGINAL INPUT
    print("TRANSPOSING... THIS WILL TAKE A WHILE")
    midiDataset = torch.unsqueeze(midiFiles[0,:,:],dim=0)
    #print(midiDataset.size())
    for i, midiFile in enumerate(midiFiles):
        #print(midiFile.size())
        tempFile = torch.unsqueeze(torchRoll(midiFile, 0, axis = 1),0)
        midiDataset = torch.cat((midiDataset,tempFile),dim=0)
        for j in range(1, 7): #FOR ALL range(0,midiFiles.shape[2])
                tempFile1 = torch.unsqueeze(torchRoll(midiFile, j, axis = 1),dim=0)
                tempFile2 = torch.unsqueeze(torchRoll(midiFile, -j, axis = 1),dim=0)
                midiDataset = torch.cat((midiDataset,tempFile1,tempFile2))
        if(i%100==0):
            print("({}/{}) {:.0f}%".format(i, midiFiles.size()[0], 100.*i/midiFiles.size()[0]))
    return midiDataset


def deleteZeroMatrices(tensor):
    # This function deletes zero matrices (sequences with no note at all)
    zeros = []
    for i,file in enumerate(tensor):
        if(np.any(file) == False):
            zeros.append(i)

    return np.delete(tensor, np.array(zeros),axis=0)
