import mido
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LiveParser():
	def __init__(self, bpm=120, ppq=24, number_seq=0, end_seq_note=127):
		self.bpm = bpm
		#pulses per quarter note
		self.ppq = ppq
		self.seconds2tick = 60. / (bpm * ppq)
		self.current_tick = -1
		self.sequence = []
		self.start_time = time.time()
		self.end_seq_note = end_seq_note
		self.bar_length = ppq * 4
		self.seq_length_ticks = self.bar_length * number_seq
		self.counter_metronome = 0
		self.metronome = 0

	def open_port(self, callback_function):
		self.in_port = mido.open_input(callback=callback_function)
		print("These input ports are available: ", mido.get_input_names())
		print("Using input port: ", self.in_port)

	def reset_clock(self):
		self.start_time = time.time()
		self.current_tick = -1
		self.metronome = 0
		self.counter_metronome = 0

	def clock(self):
		self.current_time = time.time() - self.start_time
		self.temp_tick = int(self.current_time / self.seconds2tick)
		if(self.temp_tick > self.current_tick):
			self.current_tick = self.temp_tick
			# print("clock {}".format(self.current_tick))
			if(self.current_tick % self.ppq == 0):
				self.counter_metronome += 1
		if(self.current_tick == self.seq_length_ticks):
			if self.sequence:
				return 1
			else:
				print("No note was played - starting over!\n")
				self.reset_clock()
		if(self.counter_metronome > self.metronome):
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
		print("Parsing...")
		pianoroll = np.zeros((self.seq_length_ticks, 128))

		for note in self.sequence:
			# note on range in ints (all midi channels 1-16)
			if(note[1] >= 144 and note[1] < 160):
				pianoroll[note[0],note[2]] = note[3]
			# Note off range in ints
			elif(note[1] >= 128 and note[1] < 144):
				try:
					noteOnEntry = np.argwhere(pianoroll[:note[0],note[2]])[-1][0]
					print(noteOnEntry)
				except:
					noteOnEntry = 0
				# some midi instruments send note off message with 0 or constant velocity
				# we use the velocity of the corresponding note on message
 				# TODO USE VELOCITY OF NOTE ON MESSAGE 
 				# BUGGY
				if(note[3] == 0):
					lastVelocity = pianoroll[noteOnEntry,note[2]]
					pianoroll[noteOnEntry+1:note[0]+1,note[2]] = lastVelocity	
				else:
					pianoroll[noteOnEntry+1:note[0]+1,note[2]] = note[3]

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
	midi.open_port(midi.parse_notes)
	# midi.open_port(midi.print_message)
	midi.reset_clock()
	while (True):
		status_played_notes = midi.clock()
		if status_played_notes:
			sequence = midi.parse_to_matrix()
			break

	plt.imshow(sequence.transpose(1,0), origin='lower')
	plt.show()
