# Live Interaction with VAE and LSTM

1. Connect a MIDI device (or use VMPK) to your computer.with fluidsynth

2. Use your favorite DAW or fluidsynth to synthesize the MIDI notes

3. For running the Many-to-Many (4-to-4 sequences) LSTM with live input, try:
```bash
python Predicter_Many2Many_LiveInput.py
```
Make sure you have your virtual environment running!

# Live Interaction with your own models!

TODO: Link to README which explains how to train vae and lstm

For more details on how to run this script with your own models, try:
```bash
python Predicter_Many2Many_LiveInput.py -h
```