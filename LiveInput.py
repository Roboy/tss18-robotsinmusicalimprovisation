import serial
import serial.tools.list_ports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

MIDI_BAUD_RATE = 31250

class MIDIMatrixParser():
	def __init__(self, baud_rate=31250):
		self.port = None
		self.baud_rate = baud_rate
		self.clock = 0
		self.track = []

		#automatically searches for arudino or genuino mega 
		#and sets the port
		ports = list(serial.tools.list_ports.comports())
		tempPort = []
		for p in ports:
		    #print(p[1])
		    if ("Mega") in p[1]:
		        tempPort.append(p[0])
		        print("Arduino or Genuino Mega was found! Port is:", p[0], "\n")
		#if multiple arduinos connected choose 1st arduino found
		self.port = tempPort[0]
		if(self.port == None):
			print("Arduino or Genuino Mega was not found. Please check connections!")

		#creates pyserial Serial object for given port 
		#which can be read from and written to with .read() and .write()
		self.serial = serial.Serial(self.port, self.baud_rate)
	
	def readSerial(self):
		#read first byte and decide if status byte or midi message
		s = self.serial.read()
		if(s >= b'\xf0'):
			self.statusByte(s)
		else:
			self.midiMessage(s)

	def statusByte(self, msg):
		#handles status bytes = one byte midi messages

		#clock
		if(msg == b'\xf8'):
			self.clock += 1

		#active sensing
		#if(msg == b'\xf0'):
		#	print("Active Sense")

	def midiMessage(self, msg):
		#for now just use Note On \x90-\x9F and Note Off \x80-\x8F messages for parsing
		#could be extended to support more midi messages like aftertouch or pitch bend 
		
		#Note On
		if(msg >= b'\x90' and msg < b'\xA0'):
			data1 = self.serial.read()[0]
			data2 = self.serial.read()[0]
			self.track.append([self.clock, msg[0] , data1, data2])
			print("Note On Ch.:",msg[0]%143, "Note:", data1, "Vel:", data2)
		#Note Off
		elif(msg >= b'\x80' and msg < b'\x90'):
			data1 = self.serial.read()[0]
			data2 = self.serial.read()[0]
			self.track.append([self.clock, msg[0] , data1, data2])
			print("Note Off Ch.:", msg[0]%127, "Note:", data1, "Vel:", data2)

	def parseTrack(self):
		print("Parsing...")
		pianoroll = np.zeros((self.clock, 128))

		for note in self.track:
			#print(note)
			#Note On range in ints
			if(note[1] >= 144 and note[1] < 160):
				pianoroll[note[0],note[2]] = note[3]

			#Note off range in ints
			elif(note[1] >= 128 and note[1] < 144):
				try:
					noteOnEntry = np.argwhere(pianoroll[:note[0],note[2]])[-1][0]
				except:
					noteOnEntry = 0

				#some midi instruments send note off message with 0 velocity
				if(note[3] == 0):
					lastVelocity = pianoroll[noteOnEntry,note[2]]
					pianoroll[noteOnEntry+1:note[0]+1,note[2]] = lastVelocity	
				else:
					pianoroll[noteOnEntry+1:note[0]+1,note[2]] = note[3]

		return pianoroll


midi = MIDIMatrixParser()
#sets serial with automatically set port
midi.setSerial()

#how many ticks should be recorded?
#192 ticks == 4 quarter notes (is this true??)
tickCount = 192

while(True):
	midi.readSerial()
	#temp clock prints for easier live input
	if(midi.clock == tickCount/4):
		print("1")
	if(midi.clock == tickCount/3):
		print("2")
	if(midi.clock == tickCount/2):
		print("3")
	#for prototyping break when midi clock reaches certain ticks
	if(midi.clock == tickCount):
		print("4")
		pianoroll = midi.parseTrack()
		#start over when no note was received
		if(np.any(pianoroll) == False):
			print("No note was played - starting over")
			midi.clock = 0
			continue
		break

print('')
print(pianoroll.shape)
print(np.argmax(pianoroll, axis=1))
plt.imshow(pianoroll.transpose(1,0), origin='lower')
plt.show()
