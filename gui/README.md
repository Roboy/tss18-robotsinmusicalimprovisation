## Latent space modifier
A graphical user interface (GUI) to change values in latent space of the variational autoencoder

![Variational Autoencoder GUI](/imgs/gui.png)


### How to
Navigate to this folder and make sure virtual environment is running:
```bash
python VAEsemane_GUI.py
```
#### In GUI:
1. Choose MIDI Port for input device

2. Choose VAE model you want to use

3. Run / Run Endless to start

#### Modi:

- **Run**: Interactively play with the variational autoencoder. Pressing run will start human metronome and wait for input. If input is recognized the model will respond with an improvised sequence.

- **Run Endless**: Variational autoencoder model that uses its output as input to play music endlessly. Play one sequence as input and the variational autoencoder will start improvising based on this input

- **Stop**: Stops the model

- **Randomize**: Randomize all potentiometer positions

- **Reset**: Resets all potentiometer positions

During interaction the BPM, number of bars and the temperature of the variational autoencoder can be controlled. BPM changes Tempo, number of bars chanages the amount of bars for input and ouput and temperature is a MIDI note activation threshold, so for higher temperatures more notes will be generated. Also, human metronome indicates that waiting for human input and computer metronome indicates that the computer is currently improvising.
