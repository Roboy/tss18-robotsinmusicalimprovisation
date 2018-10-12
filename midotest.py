import mido
import time

print(mido.get_input_names())
print(mido.open_input())

#beat per minute
bpm = 120

#pulses per quarter note
#96 PPQ is sufficient to capture 
#enough temporal variation according to wikipedia
#we will use 24
ppq = 24 

bar_length = ppq * 4


class LiveParser():
	def __init__(self, bpm=120, ppq=24):
		self.bpm = bpm
		self.ppq = ppq
		self.seconds2tick = 60. / (bpm * ppq)
		self.current_tick = 0
		self.track = []
		self.start_time = time.time()

	def open_port(self, callback_function):
		self.port = mido.open_input(callback=callback_function)

	def clock(self):
		self.current_time = time.time()-self.start_time
		self.temp_tick = int(self.current_time/self.seconds2tick)

	def print_message(self, msg):
		print(msg)

	def print_message_bytes(self, msg):
		print(msg.bytes())

	def parse_notes(self, message):
		msg = message.bytes()
		self.track.append([self.current_tick, msg[0], msg[1], msg[2]])
		print(self.track)




midi = LiveParser(bpm=bpm, ppq=ppq)
midi.open_port(midi.parse_notes)


while (True):
	current_time = time.time()-midi.start_time
	temp_tick = int(current_time/midi.seconds2tick)
	if (temp_tick > midi.current_tick):
		midi.current_tick = temp_tick
		print("clock {}".format(midi.current_tick))
	if(midi.current_tick == bar_length*1):
		if not midi.track:
			print("No note played starting over!")
		midi.start_time = time.time()
		midi.current_tick = 0
	
print(midi.track)

