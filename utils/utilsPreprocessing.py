import numpy as np
import torch
import torch.utils.data as data
import pypianoroll as ppr
import music21
import glob
import time


def getSlicedPianorollMatrixTorch(pathToFile, binarize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqLength = 96

    track = ppr.Multitrack(pathToFile, beat_resolution=24)

    #HOW TO FIND OUT IF MULTITRACK OBJECT IS DRUM???????????
    #print(track.tracks[0])

    #downbeats = track.get_downbeat_steps()
    #print(downbeats)
    track = track.get_stacked_pianoroll()

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
    track = track.get_stacked_pianoroll()

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


def getSlicedPianorollMatrixList(pathToFile, binarize=True, beat_resolution=24):

    seqLength = 96

    track = ppr.Multitrack(pathToFile, beat_resolution=beat_resolution)
    #downbeats = track.get_downbeat_steps()
    #print(downbeats)
    track = track.get_stacked_pianoroll()

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
        #print(track.shape)
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


def pianorollMatrixToTempMidi(matrix, path='../tempMidiFiles/temp.mid', prediction=True,
    show=False, showPlayer=False, autoplay=False):
    # matrix must be of LENGTHxPITCH dimension here: (96 or more,128)

    ###THIS IS A WORKAROUND SO NO NEW NOTES ARE SET ON THE LAST 2 TICKS
    ###IF NOTE IS SET IT WILL RAISE AN ERROR THAT THERE CANNOT BE A BEGIN AND
    ###END ON 4.0/4.0 (measures)
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
    # This function deletes zero matrices (sequences with no note at all)
    zeros = []
    for i,file in enumerate(tensor):
        if(np.any(file) == False):
            zeros.append(i)

    return np.delete(tensor, np.array(zeros),axis=0)


class createDatasetAE(data.Dataset):
    def __init__(self, file_path, beat_res=24, seq_length=96, binarize=True, bars=1):
        self.all_files = glob.glob(file_path + '*.mid')
        self.beat_res = beat_res
        self.bars = bars
        self.binarize = binarize
        self.seq_length = seq_length

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        try:
            # load song from midi files and parse to numpy
            track = ppr.Multitrack(self.all_files[idx], beat_resolution=self.beat_res)
            # print(track)
            track = track.get_stacked_pianoroll()
            # print(track.shape)
            # if: 1 track midifile
            # else: quick fix for multitrack, melody in almost every song on midi[0]
            if track.shape[2]==1:
                track = np.squeeze(track,2)
            else:
                track = track[:,:,0]

            # if length differs from seq_length, cut it to self.seq_length
            if track.shape[0] > self.seq_length:
                track = track[:4*self.beat_res*self.bars]
            elif track.shape[0] < self.seq_length:
                pad_with = self.seq_length - track.shape[0]
                # print("pad_with = {}".format(pad_with))
                temp = np.zeros((pad_with, 128), dtype=np.uint8)
                # print("temp.shape = {}".format(temp.shape))
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
            # print("sequence.shape = {}".format(sequence.shape))
            # print("sequence.dtype = {}".format(sequence.dtype))
        except:
            print("MIDI file warning. Skipped a MIDI file because was not working properly.")
            sequence = np.zeros((1, self.seq_length, 60), dtype=np.uint8)
        return torch.from_numpy(sequence)



######################################################################
###############LSTM PREPROCESSING#####################################
######################################################################

def padPianoroll(pianoroll, max_length, pad_value=0):
    org_length, all_pitches = pianoroll.shape
    padded_pianoroll = np.zeros((max_length, all_pitches))
    padded_pianoroll[:] = pad_value
    padded_pianoroll[:org_length,:] = pianoroll

    return padded_pianoroll


class createDatasetLSTM(data.Dataset):
    def __init__(self, pathToFiles, beat_res=4, binarize=True, seq_length=16,
                    force_length=False, force_value=16):
        self.all_files = glob.glob(pathToFiles)
        self.beat_res = beat_res
        self.binarize = binarize
        self.seq_length = seq_length
        self.max_length = 0
        self.force_length = force_length
        self.force_value = force_value

    def setMaxLength(self):
        if(self.force_length):
            self.max_length = self.force_value
        else:
            self.max_length = 0
            for f in self.all_files:
                temp_length = ppr.Multitrack(f, beat_resolution=self.beat_res).get_max_length()
                if(temp_length > self.max_length):
                    self.max_length = temp_length
        print('Longest sequences contains {} ticks'.format(self.max_length))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        track = ppr.Multitrack(self.all_files[idx], beat_resolution=self.beat_res)
        track = track.get_stacked_pianoroll()

        #binarize
        if(self.binarize):
            track[track > 0] = 1

        #IF 1 TRACK MIDIFILE
        if(track.shape[2]==1):
            track = np.squeeze(track,2)
        #quick fix for multitrack
        else:
            track = track[:,:,0]
        #transpose notes below and above playing range
        track = transposeNotesHigherLower(track)
        #cut octaves to 60 pitches
        track = cutOctaves(track)

        #add class for rests
        new_track = np.zeros((track.shape[0],track.shape[1]+1))
        new_track[:,:track.shape[1]] = track
        for i, tick in enumerate(new_track):
            #note in time step
            if(np.any(tick)):
                continue
            #no note in time step == rest (class 61)
            else:
                new_track[i,-1] = 1

        if(self.force_length):
            seq_length = self.force_value -1
            input_pianoroll = new_track[:seq_length,:]
            ground_truth_pianoroll = new_track[1:self.force_value,:]
        else:
            seq_length = new_track.shape[0]-1

            input_pianoroll = new_track[:-1,:]
            ground_truth_pianoroll = new_track[1:,:]

            input_pianoroll = padPianoroll(input_pianoroll, self.max_length, pad_value=0)
            ground_truth_pianoroll = padPianoroll(ground_truth_pianoroll, self.max_length,
                pad_value=-100)

        return input_pianoroll, ground_truth_pianoroll, seq_length



"""
class createSeqDatasetLSTM(data.Dataset):
    def __init__(self, pathToFiles, seq_length=16, beat_res=4, binarize=True):
        self.all_files = glob.glob(pathToFiles)
        self.beat_res = beat_res
        self.binarize = binarize
        self.seq_length = seq_length
        self.max_length = 0

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        track = ppr.Multitrack(self.all_files[idx], beat_resolution=self.beat_res)
        track = track.get_stacked_pianoroll()

        cut_length = self.seq_length+1

        #binarize
        if(self.binarize):
            track[track > 0] = 1

        length_temp = track.shape[0]
        print(length_temp)

        if(length_temp % cut_length != 0):
            track = track[:-(length_temp % cut_length), :]
        length = track.shape[0]
        print(length)
        #IF 1 TRACK MIDIFILE
        if(track.shape[2]==1):
            track = np.squeeze(track,2)
            track = np.array(np.split(track, int(length/cut_length),axis=0))
        #quick fix for multitrack
        else:
            track = track[:,:,0]
            track = np.array(np.split(track, int(length/cut_length),axis=0))

        track = transposeNotesHigherLower(track)
        track = cutOctaves(track)

        print(track.shape)

        seq_length = track.shape[1]-1

        input_pianoroll = track[:,:-1,:]
        ground_truth_pianoroll = track[:,1:,:]

        #input_pianoroll = padPianoroll(input_pianoroll, self.max_length, pad_value=0)
        #ground_truth_pianoroll = padPianoroll(ground_truth_pianoroll, self.max_length,
        #    pad_value=-100)

        return input_pianoroll, ground_truth_pianoroll, seq_length


"""
