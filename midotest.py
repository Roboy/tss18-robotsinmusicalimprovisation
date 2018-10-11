import mido
import time

print(mido.get_input_names())
print(mido.open_input())

start_time = time.time()

with mido.open_input() as in_port:
    for msg in in_port:
    	note_play_time = time.time()-start_time
    	print(note_play_time.second2tick(), msg.bytes())
