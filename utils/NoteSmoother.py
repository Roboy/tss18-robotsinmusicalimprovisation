import numpy as np


class Note:
    def __init__(self, note, begin):
        self.note = note
        self.begin = begin
        self.end = None
        self.length = None

    def __str__(self):
        return 'Note = {} at Tick {}'.format(self.note, self.begin, self.end)

    def __repr__(self):
        return 'Midi On: Note = {} at Tick = {} and ends at\
                            {} with length {}\n'.format(self.note,
                                        self.begin, self.end, self.length)

    def compute_end(self, sequence):
        end = np.argwhere(sequence[self.begin:,self.note] == 0)
        if end.any():
            self.end = self.begin + end[0][0]
        else:
            self.end = sequence.shape[0]
        self.length = self.end - self.begin


class NoteSmoother:
    def __init__(self, sequence, threshold=1):
        self.sequence = sequence
        self.threshold = threshold

    def smooth(self):
        note_on = []
        for i in range(self.sequence.shape[0]):
            if i==0:
                current_note = np.argwhere(self.sequence[0] > 0)
                for current in current_note:
                    note_on.append(Note(current[0], i))
            else:
                current_note = np.argwhere(self.sequence[i] > 0)
                for current in current_note:
                    if current not in np.argwhere(self.sequence[i-1] > 0):
                        note_on.append(Note(current[0], i))
        for note in note_on:
            note.compute_end(self.sequence)
            if note.length <= self.threshold:
                self.sequence[note.begin:note.end, note.note] = 0

        return self.sequence
