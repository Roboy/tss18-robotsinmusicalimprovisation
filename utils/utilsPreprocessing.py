import numpy as np
import torch
import pypianoroll as ppr
import music21


def getSlicedPianorollMatrixTorch(pathToFile, binarize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqLength = 96
    
    track = ppr.Multitrack(pathToFile, beat_resolution=24)

    #HOW TO FIND OUT IF MULTITRACK OBJECT IS DRUM???????????
    #print(track.tracks[0])

    #downbeats = track.get_downbeat_steps()
    #print(downbeats)
    track = track.get_stacked_pianorolls()

    """BINARIZE"""
    if(binarize):
        track[track > 0] = 1 
    
    track = torch.from_numpy(track).byte()
    #print(track.size())

    """#DELETE LAST ROWS IN TIME AXIS TO MATCH DIMENSION %96
    ##### BETTER SOLUTION THAN CUTTING OFF THE END OF THE TRACK ???"""
    lengthTemp = track.shape[0]
    #print(lengthTemp%seqLength)
    if(lengthTemp%seqLength != 0):
        track = track[:-(lengthTemp%seqLength), :]
    length = track.shape[0]

    #IF 1 TRACK MIDIFILE
    if(track.shape[2]==1):
        track = track.permute(2,0,1)
        track = torch.chunk(track, int(length/seqLength),dim=1)
        return torch.cat(track,dim=0)
    
    #ELSE MULTITRACK MIDIFILE
    else:
        endTrack = torch.chunk(track[:,:,0].unsqueeze(0),int(length/seqLength),dim=1)
        endTrack = torch.cat(endTrack,dim=0)
        for i in range(1,track.shape[2]):
            track1 = track[:,:,i]
            temp = torch.chunk(track1.unsqueeze(0), int(length/seqLength),dim=1)
            temp = torch.cat(temp,dim=0)
            endTrack = torch.cat((endTrack,temp),dim=0)        
        return endTrack

def getSlicedPianorollMatrixNp(pathToFile, binarize=True):
    
    seqLength = 96
    
    track = ppr.Multitrack(pathToFile, beat_resolution=24)
    #downbeats = track.get_downbeat_steps()
    #print(downbeats)
    track = track.get_stacked_pianorolls()

    """BINARIZE"""
    if(binarize):
        track[track > 0] = 1 

    """#DELETE LAST ROWS IN TIME AXIS TO MATCH DIMENSION %96
    ##### BETTER SOLUTION THAN CUTTING OFF THE END OF THE TRACK ???"""
    lengthTemp = track.shape[0]
    #print(lengthTemp%seqLength)
    if(lengthTemp%seqLength != 0):
        track = track[:-(lengthTemp%seqLength), :]
    length = track.shape[0]

    #IF 1 TRACK MIDIFILE
    if(track.shape[2]==1):
        track = np.squeeze(track,2)
        return np.array(np.split(track, int(length/seqLength),axis=0))
    
    #ELSE MULTITRACK MIDIFILE
    else:
        endTrack = np.array(np.split(track[:,:,0], int(length/seqLength),axis=0))
        for i in range(1,track.shape[2]):
            track1 = track[:,:,i]
            temp = np.array(np.split(track1, int(length/seqLength),axis=0))
            endTrack = np.concatenate((endTrack,temp))
                
    return endTrack

def getSlicedPianorollMatrixList(pathToFile, binarize=True):
    
    seqLength = 96
    
    track = ppr.Multitrack(pathToFile, beat_resolution=24)
    #downbeats = track.get_downbeat_steps()
    #print(downbeats)
    track = track.get_stacked_pianorolls()

    """BINARIZE"""
    if(binarize):
        track[track > 0] = 1 
    #print(track.dtype)
    #print(track.shape)

    """#DELETE LAST ROWS IN TIME AXIS TO MATCH DIMENSION %96
    ##### BETTER SOLUTION THAN CUTTING OFF THE END OF THE TRACK ???"""
    lengthTemp = track.shape[0]
    #print(lengthTemp%seqLength)
    if(lengthTemp%seqLength != 0):
        track = track[:-(lengthTemp%seqLength), :]
    length = track.shape[0]

    #IF 1 TRACK MIDIFILE
    if(track.shape[2]==1):
        track = np.squeeze(track,2)
        track = np.split(track, int(length/seqLength),axis=0)
        #print(len(track))
        #print(track)
        
        return track
    
    #ELSE MULTITRACK MIDIFILE
    else:
        endTrack = []
        for i in range(track.shape[2]):
            track1 = track[:,:,i]
            temp = np.split(track1, int(length/seqLength),axis=0)

            for temp2 in temp:
                endTrack.append(temp2)
         
    return endTrack


def transposeNotesHigherLower(a):
    #THIS FUNCTION TRANSPOSES ALL NOTES HIGHER THAN B6 AND LOWER THAN C2
    #INTO 5 OCTAVE RANGE FROM C2 TO B6
    for i,track in enumerate(a):
        #print(track.shape)
        octCheck = np.argwhere(track>0)
        for j in octCheck:
            if(j[1]>=36 and j[1]<96):
                break
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
                #print(j[0])
                #print(j[1])
                a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],-12)
            elif(j[1]>=108 and j[1]<120):
                #print('There is a note above 108')
                a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],-24)
            elif(j[1]>=120):
                #print('There is a note above 120')
                a[i:i+1,j[0]:j[0]+1,:] = np.roll(a[i:i+1,j[0]:j[0]+1,:],-36)

    return a

def cutOctaves(tensor): 
    tensor = tensor[:,:,36:-32]
    return tensor


def addCuttedOctaves(matrix):
    if(matrix.shape[1]!=128):
        if(matrix.ndim==3):
            matrix = np.pad(matrix,[[0,0],[0,0],[36,32]],'constant')
        else:
            matrix = np.pad(matrix,[[0,0],[36,32]],'constant')
    return matrix

def pianorollMatrixToTempMidi(matrix, path='tempMidiFiles/temp.mid', prediction=True,
    show=True, showPlayer=True, autoplay=False): 
    #MATRIX MUST BE OF DIMENSION LENGTHxPITCH here: (96,128)
 
    ###THIS IS A WORKAROUND SO NO NEW NOTES ARE SET ON THE LAST 2 TICKS
    ###IF NOTE IS SET IT WILL RAISE AN ERROR THAT THERE CANNOT BE A BEGIN AND 
    ###END ON 4.0/4.0 (measures)
    if(prediction):
        matrix[-2:,:] = 0

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

    ###TRY EXCEPT TAKES LONGER BUT WORKS BETTER
    """
    try:    
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
    except:
        matrix[-2:,:] = 0
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
    """
def debinarizeMidi(a, prediction=True,velocity=127):
    if(prediction):
        ###MONOPHONIC###
        #rowMax = a.max(axis=1).reshape(-1, 1)
        #a[:] = np.where((a == rowMax) & (a > 0), velocity, 0)

        ###POLYPHONIC###
        #print("POLYPHONIC")
        a[:] = np.where((a > 0), velocity, 0)
    elif(prediction==False):
        a[:] = np.where(a == 1,velocity,0)
    return a

def noteThreshold(a, threshold=0.5, velocity=127):
    a[:] = np.where(a > threshold, velocity, 0)

    return a
    
def torchRoll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)
    
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
    #DELETE ZERO MATRICES (SEQUENCES WITH NO NOTE AT ALL)
    #print(tensor.shape)
    zeros = []
    for i,file in enumerate(tensor):
        #print(file.size())
        if(np.any(file) == False):
            zeros.append(i)

    return np.delete(tensor, np.array(zeros),axis=0)

def checkActivations(matrix):
    #CHECK IF MIDI ACTIVATION THRESHOLD IS REACHED
    #INCLUDES RESTS BY USING AN ACTIVATION THRESHOLD FOR A MIDI NOTE
    #BY LOOKING AT THE MATRICES SET IT TO VALUE FOR NOW
    """HARDCODED 0.7 --> DYANMIC THRESHOLD BY NORMALIZING PREDICTION?"""
    activationCheck = np.sum(matrix, axis=1)
    #print(activationCheck)
    activationCheck = np.where(activationCheck < 7)[0].tolist()
    matrix[activationCheck,:] = 0
    
    return matrix