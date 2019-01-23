import sys
import os
import numpy as np
import torch
import mido
import torch.utils.data
from utils.utils import (cutOctaves, debinarizeMidi, addCuttedOctaves)
from VAE.VAE_Reconstruct_TrainNEW import VAE
from loadModel import loadModel, loadStateDict
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSpacerItem, \
    QVBoxLayout, QWidget, QDial, QPushButton, QGroupBox, QGridLayout, QMainWindow, \
    QAction, qApp, QFileDialog

from LIVE.LiveInput_ClassCompliant import LiveParser


def vae_main(live_instrument, model):
    # reset live input clock
    print("\nUser input\n")
    live_instrument.reset_sequence()
    live_instrument.reset_clock()
    while True:
        status_played_notes = live_instrument.clock()
        if status_played_notes:
            sequence = live_instrument.parse_to_matrix()
            live_instrument.reset_sequence()
            break

    # send live recorded sequence through model and get improvisation
    with torch.no_grad():
        sample = np.array(np.split(sequence, 1))

        # prepare sample for input
        sample = cutOctaves(sample)
        sample = torch.from_numpy(sample).float().to(device)
        sample = torch.unsqueeze(sample,1)

        # model
        mu, logvar = model.encoder(sample)

        # TODO reparameterize to get new sequences here with GUI??

        #reconstruction, soon ~prediction
        pred = model.decoder(mu)

        # reorder prediction
        pred = pred.squeeze(1)
        prediction = pred[0]

        # TODO TEMP for more sequences
        if pred.size(0) > 1:
            for p in pred[1:]:
                prediction = torch.cat((prediction, p), dim=0)

        prediction = prediction.cpu().numpy()
        # normalize predictions
        prediction /= np.abs(np.max(prediction))

        # check midi activations to include rests
        prediction[prediction < (1 - 0.7)] = 0
        prediction = debinarizeMidi(prediction, prediction=True)
        prediction = addCuttedOctaves(prediction)

        print("\nPrediction\n")
        # print(prediction)
        # play predicted sequence note by note
        live_instrument.reset_clock()
        play_tick = -1
        old_midi_on = np.zeros(1)
        played_notes = []
        while True:
            done = live_instrument.computer_clock()
            if live_instrument.current_tick > play_tick:
                play_tick = live_instrument.current_tick
                midi_on = np.argwhere(prediction[play_tick] > 0)
                if midi_on.any():
                    for note in midi_on[0]:
                        if note not in old_midi_on:
                            current_vel = int(prediction[live_instrument.current_tick,note])
                            # print(current_vel)
                            live_instrument.out_port.send(Message('note_on',
                                note=note, velocity=current_vel))
                            played_notes.append(note)
                else:
                    for note in played_notes:
                        live_instrument.out_port.send(Message('note_off',
                                                    note=note))#, velocity=100))
                        played_notes.pop(0)

                if old_midi_on.any():
                    for note in old_midi_on[0]:
                        if note not in midi_on:
                            live_instrument.out_port.send(Message('note_off', note=note))
                old_midi_on = midi_on

            if done:
                break

    live_instrument.reset_sequence()


class Poti(QWidget):
    def __init__(self, mean, variance, parent=None):
        super(Poti, self).__init__(parent=parent)
        # self.verticalLayout = QVBoxLayout()
        # self.horizontalLayout = QHBoxLayout()
        self.label = QLabel(self)
        self.dial = QDial(self)
        # self.horizontalLayout.addWidget(self.dial)
        # self.horizontalLayout.addWidget(self.label)
        # self.verticalLayout.addLayout(self.horizontalLayout)

        self.current_value = None
        self.dial.valueChanged.connect(self.setLabelValue)
        self.mean = mean
        self.variance = variance
        self.dial.setMinimum(-100)
        self.dial.setMaximum(100)
        self.minimum = mean - variance
        self.maximum = mean + variance
        self.setLabelValue(self.dial.value())
        self.dial.setSliderPosition(0)

    def setLabelValue(self, value):
        # TODO label with real values from mean and variance?
        # self.current_value = self.minimum + (float(value) /
        #     (self.dial.maximum() - self.dial.minimum())) * (self.maximum - self.minimum)
        self.label.setText("{}%".format(self.dial.value()))

    def updatePoti(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def reset(self):
        self.dial.setSliderPosition(0)

    def randomPos(self):
        rand = np.random.randint(self.dial.minimum(),self.dial.maximum())
        self.dial.setSliderPosition(rand)


class VAE_Window(QWidget):
    def __init__(self, parent=None):
        super(VAE_Window, self).__init__(parent=parent)
        self.title = 'VAEsemane'
        self.left = 20
        self.top = 20
        self.width = 800
        self.height = 600
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # widgets
        self.metronome = QLabel(self)
        self.b_run = QPushButton('Run VAE')
        self.b_reset = QPushButton('Reset Potis')
        self.b_random = QPushButton('Randomize Potis')
        self.potis = [Poti(0.,10.) for i in range(100)]
        # connect widgets
        self.b_run.clicked.connect(self.run_vae)
        self.b_reset.clicked.connect(self.reset_click)
        self.b_random.clicked.connect(self.random_click)

        # layout
        self.createGridLayout()
        windowLayout = QVBoxLayout()
        # windowLayout.addWidget(self.metronome)
        windowLayout.addWidget(self.horizontalGroupBox)
        windowLayout.addWidget(self.b_run)
        windowLayout.addWidget(self.b_reset)
        windowLayout.addWidget(self.b_random)
        self.setLayout(windowLayout)

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox()
        layout = QGridLayout()
        for i in range(10):
            for j in range(10):
                layout.addWidget(self.potis[i*10+j],i,j)
        self.horizontalGroupBox.setLayout(layout)

    def reset_click(self):
        for poti in self.potis:
            poti.reset()

    def random_click(self):
        for poti in self.potis:
            poti.randomPos()

    def run_vae(self):
        # TODO THREAD THIS
        model = VAE()
        model = loadStateDict(model, self.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("This should run VAE")
        # while True:
        #     vae_main(self.live_instrument, model)



    def set_metronome(self):
        pass

    def update_tick(self):
        pass

    def update(self):
        pass

    def set_instrument(self, q):
        self.live_instrument = LiveParser(port=q.text())
        self.live_instrument.open_inport(self.live_instrument.parse_notes)
        self.live_instrument.open_outport()

    def set_model_path(self, file_path):
        self.model_path = file_path
        # print(self.model_path)

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()


        self.init_ui()

    def init_ui(self):
        # cerate menu bar
        bar = self.menuBar()
        bar.setNativeMenuBar(False)
        # first tab file
        file_tab = bar.addMenu('File')
        vae_action = QAction('&VAEsemane', self)
        file_tab.addAction(vae_action)
        vae_action.triggered.connect(self.start_vae)
        lstm_action = QAction('&LSTM', self)
        file_tab.addAction(lstm_action)
        lstm_action.triggered.connect(self.start_lstm)
        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('Ctrl+W')
        file_tab.addAction(quit_action)
        quit_action.triggered.connect(self.quit_trigger)

        # second tab midi port, sets instrument
        midi_port_tab = bar.addMenu('MIDI Port')
        available_port = mido.get_input_names()
        midi_ports = []
        midi_actions = []
        for port in available_port:
            midi_actions.append(QAction(port, self))
            midi_port_tab.addAction(midi_actions[-1])
        midi_port_tab.triggered.connect(self.set_instrument)

        # third tab model
        model_tab = bar.addMenu('Model')
        model_find_act = QAction('&Find', self)
        model_tab.addAction(model_find_act)
        model_find_act.triggered.connect(self.model_find_func)




        self.show()

    def quit_trigger(self):
        qApp.quit()

    def start_vae(self):
        self.form_widget = VAE_Window()
        self.setCentralWidget(self.form_widget)

    def start_lstm(self):
        self.form_widget = QLabel('LSTM')
        self.setCentralWidget(self.form_widget)

    def set_instrument(self, q):
        VAE_Window.set_instrument(self.form_widget, q)

    def model_find_func(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('..'))
        VAE_Window.set_model_path(self.form_widget, filename[0])



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = GUI()
    w.show()

    sys.exit(app.exec_())
