import mido
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class LiveParser():
    def __init__(self, port=None, bpm=120, ppq=24, number_seq=0, end_seq_note=127):
        self.bpm = bpm  # beat per minute
        self.ppq = ppq  # pulses per quarter note
        self.seconds2tick = 60. / (bpm * ppq)  #seconds to tick conversion
        self.current_tick = -1
        self.sequence = []
        self.start_time = time.time()
        self.end_seq_note = end_seq_note
        self.bar_length = ppq * 4
        self.seq_length_ticks = self.bar_length * number_seq
        self.counter_metronome = 0
        self.metronome = 0
        self.port = port
        self.current_time = 0.
        self.temp_tick = 0
        self.out_port = None

    def open_inport(self, callback_function):
        avail_ports = mido.get_input_names()
        ports_dict = {i: avail_ports[i] for i in range(len(avail_ports))}
        print("These input ports are available: ", ports_dict)
        if self.port == None:
            port_num = int(input("Which port would you like to use? "))
            self.in_port = mido.open_input(ports_dict[port_num], callback=callback_function)
        else:
            self.in_port = mido.open_input(port, callback=callback_function)
        print("Using input port: ", self.in_port)

    def open_outport(self):
        # TODO autoroute midi port to virtual synth possible??
        avail_out_ports = mido.get_output_names()
        ports_dict = {i: avail_out_ports[i] for i in range(len(avail_out_ports))}
        port = None
        for i in range(len(avail_out_ports)):
            if "Synth input" in ports_dict[i]:  # Better way than looking for this string?
                port = ports_dict[i]
        if port:
            self.out_port = mido.open_output(port)
            print("Found FLUID Synth and autoconnected!")
        else:
            self.out_port = mido.open_output("Robot port", virtual=True)
            print("Could not find FLUID Synth, created virtual midi port called 'Robot port'")
        # import pdb
        # pdb.set_trace()

    def reset_clock(self):
        self.start_time = time.time()
        self.current_tick = -1
        self.metronome = 0
        self.counter_metronome = 0

    def reset_sequence(self):
        self.sequence = []

    def clock(self):
        self.current_time = time.time() - self.start_time
        self.temp_tick = int(self.current_time / self.seconds2tick)
        if self.temp_tick > self.current_tick:
            self.current_tick = self.temp_tick
            # print("clock {}".format(self.current_tick))
            if self.current_tick % self.ppq == 0:
                self.counter_metronome += 1
        if self.current_tick == self.seq_length_ticks-1:
            if self.sequence:
                return 1
            else:
                print("No note was played - starting over!\n")
                self.reset_clock()
        if self.counter_metronome > self.metronome:
            self.metronome = self.counter_metronome
            print(self.metronome)

    def computer_clock(self):
        self.current_time = time.time() - self.start_time
        self.temp_tick = int(self.current_time / self.seconds2tick)
        if self.temp_tick > self.current_tick:
            self.current_tick = self.temp_tick
            # print("clock {}".format(self.current_tick))
            if self.current_tick % self.ppq == 0:
                self.counter_metronome += 1
        if self.current_tick == self.seq_length_ticks-1:
            return 1
        if self.counter_metronome > self.metronome:
            self.metronome = self.counter_metronome
            print(self.metronome)

    def print_message(self, msg):
        print(msg)

    def print_message_bytes(self, msg):
        print(msg.bytes())

    def parse_notes(self, message):
        # print(message)
        msg = message.bytes()
        self.sequence.append([self.current_tick, msg[0], msg[1], msg[2]])

    def parse_to_matrix(self):
        # print("Parsing...")
        pianoroll = np.zeros((self.seq_length_ticks, 128))

        for note in self.sequence:
            # print(note)
            # note on range in ints (all midi channels 1-16)
            if(note[1] >= 144 and note[1] < 160):
                pianoroll[note[0]-1, note[2]] = note[3]
            # note off range in ints (all midi channels 1-16)
            elif(note[1] >= 128 and note[1] < 144):
                try:
                    noteOnEntry = np.argwhere(pianoroll[:note[0],note[2]])[-1][0]
                    # print(noteOnEntry)
                except:
                    noteOnEntry = 0
                # some midi instruments send note off message with 0 or constant velocity
                # use the velocity of the corresponding note on message
                # TODO USE VELOCITY OF NOTE ON MESSAGE
                # BUGGY, throws error if you play a note on the last midi tick of the sequence
                if note[3] == 0:
                    last_velocity = pianoroll[noteOnEntry, note[2]]
                    pianoroll[noteOnEntry+1:note[0]+1, note[2]] = last_velocity
                else:
                    pianoroll[noteOnEntry+1:note[0]+1, note[2]] = note[3]

        return pianoroll


if __name__ == '__main__':
    # beats per minute
    bpm = 120

    # pulses per quarter note
    # 96 PPQ is sufficient to capture enough temporal variation according to wikipedia
    # we will use 24
    ppq = 24

    #4/4
    bar_length = ppq * 4

    # how many bars would you like to record?
    number_seq = 2

    midi = LiveParser(bpm=bpm, ppq=ppq, number_seq=number_seq, end_seq_note=127)
    midi.open_inport(midi.parse_notes)
    # midi.open_inport(midi.print_message)
    midi.open_outport()
    midi.reset_clock()
    while (True):
        status_played_notes = midi.clock()
        if status_played_notes:
            sequence = midi.parse_to_matrix()
            break

    plt.imshow(sequence.transpose(1,0), origin='lower')
    plt.show()
