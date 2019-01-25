import sys
import os
import numpy as np
import torch
import mido
import threading
import torch.utils.data
from utils.utils import (cutOctaves, debinarizeMidi, addCuttedOctaves)
from VAE.VAE_Reconstruct_TrainNEW import VAE
from loadModel import loadModel, loadStateDict
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSpacerItem, \
    QVBoxLayout, QWidget, QDial, QPushButton, QGroupBox, QGridLayout, QMainWindow, \
    QAction, qApp, QFileDialog

from LIVE.LiveInput_ClassCompliant import LiveParser


def vae_main(live_instrument, model, device):
    while True:# reset live input clock
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


            # reparameterize with variance
            dial_vals = []
            for poti in w.form_widget.potis:
                dial_vals.append(poti.dial.value())
            dial_tensor = torch.FloatTensor(dial_vals)/100.
            print(dial_tensor)

            new = mu + (dial_tensor * logvar.exp())

            #reconstruction, soon ~prediction
            pred = model.decoder(new)

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

            # play predicted sequence note by note
            print("\nPrediction\n")
            live_instrument.computer_play(prediction=prediction)

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
        # self.title = 'VAEsemane'
        self.left = 20
        self.top = 20
        self.width = 800
        self.height = 600
        self.init_window()

    def init_window(self):
        # self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # widgets
        self.human_metronome = QLabel(str(0), self)
        self.computer_metronome = QLabel(str(0), self)
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
        self.createMetronomeLayout()
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.h_metro_box)
        windowLayout.addWidget(self.h_poti_box)
        self.adjustSize()
        windowLayout.addWidget(self.b_run)
        windowLayout.addWidget(self.b_reset)
        windowLayout.addWidget(self.b_random)
        self.setLayout(windowLayout)

    def createGridLayout(self):
        self.h_poti_box = QGroupBox()
        layout = QGridLayout()
        for i in range(10):
            for j in range(10):
                layout.addWidget(self.potis[i*10+j],i,j)
        self.h_poti_box.setLayout(layout)

    def createMetronomeLayout(self):
        self.h_metro_box = QGroupBox()
        layout = QGridLayout()
        layout.addWidget(self.human_metronome,0,0)
        layout.addWidget(self.computer_metronome,0,1)
        self.h_metro_box.setLayout(layout)

    def reset_click(self):
        for poti in self.potis:
            poti.reset()

    def random_click(self):
        for poti in self.potis:
            poti.randomPos()

    def run_vae(self):
        vae_thread = threading.Thread(target=vae_main,
                        args=(self.live_instrument, self.model, self.device))
        vae_thread.start()
        # vae_thread.join()

    def update_metronome(self):
        while(True):
            if self.live_instrument.human:
                self.computer_metronome.setText("0")
                self.human_metronome.setText("{}".format(self.live_instrument.metronome))
            else:
                self.human_metronome.setText("0")
                self.computer_metronome.setText("{}".format(self.live_instrument.metronome))


    def start_metronome(self):
        metronome_thread = threading.Thread(target=self.update_metronome)
        metronome_thread.start()

    def update_tick(self):
        pass

    def update(self):
        pass

    def set_instrument(self, q):
        self.live_instrument = LiveParser(port=q.text(), ppq=12,
                                            number_seq=1, bpm=120)
        self.live_instrument.open_inport(self.live_instrument.parse_notes)
        self.live_instrument.open_outport()
        self.start_metronome()

    def set_model_path(self, file_path):
        self.model = VAE()
        self.moodel = loadStateDict(self.model, file_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model.train():
            self.model.eval()

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()

        self.init_ui()

    def init_ui(self):
        # cerate menu bar
        self.setWindowTitle('RIMI')
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
        self.form_widget.set_model_path(filename[0])



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = GUI()
    w.show()
    sys.exit(app.exec_())
