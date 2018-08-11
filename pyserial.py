import serial

def midiNoteOff(s):
	print("MidiNoteOff")

def midiNoteOn(s):
	print("MidiNoteOn")

def Clock(s):
	global clock
	print("Clock {}".format(clock))
	clock += 1
	clock %= 256 #emulates byte behaviour

def ActiveSensing(s):
	#print("ActiveSense")
	return 0

switcher = {
		b'\x82': midiNoteOff,
		b'\x92': midiNoteOn,
		b'\xf8': Clock,
		b'\xfe': ActiveSensing
	}


ser = serial.Serial('/dev/cu.usbmodem142421', 115200)#, timeout=0)
clock = 0
print(ser.name)
while(True):
	s = ser.read()
	if(s):
		print(s)
		#s_temp = s
		#try:
		#	switcher[s](s)
		#except:
		#	print(s)

		

"""

	if(s == b'\xf8'): #f8: Clock tick
		print("Clock {}".format(clock))
		clock+=1

	#if(s):	
	#	print(s)
	
midiDict = {b'\xf8':Clock}
"""
"""
def switch_demo(argument):
    switcher = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    print switcher.get(argument, "Invalid month")
"""