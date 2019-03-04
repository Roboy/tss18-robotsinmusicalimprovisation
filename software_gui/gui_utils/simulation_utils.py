import torch
import numpy as np
from utils.utils import (cutOctaves, debinarizeMidi, addCuttedOctaves)
from utils.NoteSmoother import NoteSmoother
# import rospy


def vae_interact(gui):
    live_instrument = gui.live_instrument
    model = gui.model
    device = gui.device
    dials = gui.dials
    while True:
        print("\nUser input\n")
        # reset live input clock and prerecorded sequences
        live_instrument.reset_sequence()
        live_instrument.reset_clock()
        while True:
            status_played_notes = live_instrument.clock()
            if status_played_notes:
                sequence = live_instrument.parse_to_matrix()
                live_instrument.reset_sequence()
                break
            if not gui.is_running:
                break
        if not gui.is_running:
            break

        # send live recorded sequence through model and get response
        with torch.no_grad():
            # prepare sample for input
            sample = np.array(np.split(sequence, live_instrument.bars))
            sample = cutOctaves(sample)
            sample = torch.from_numpy(sample).float().to(device)
            sample = torch.unsqueeze(sample,1)

            # encode
            mu, logvar = model.encoder(sample)

            # reparameterize with variance
            dial_vals = []
            for dial in dials:
                dial_vals.append(dial.value())
            dial_tensor = torch.FloatTensor(dial_vals)/100.
            new = mu + (dial_tensor * 0.5 * logvar.exp())
            pred = model.decoder(new).squeeze(1)

            # for more than 1 sequence
            prediction = pred[0]
            if pred.size(0) > 1:
                for p in pred[1:]:
                    prediction = torch.cat((prediction, p), dim=0)

            # back to cpu and normalize
            prediction = prediction.cpu().numpy()
            prediction /= np.abs(np.max(prediction))

            # check midi activations to include rests
            prediction[prediction < (1 - gui.slider_temperature.value()/100.)] = 0
            prediction = debinarizeMidi(prediction, prediction=True)
            prediction = addCuttedOctaves(prediction)
            smoother = NoteSmoother(prediction, threshold=2)
            prediction = smoother.smooth()

            # play predicted sequence note by note
            if gui.chx_simulate_robot.isChecked():
                print("\nPublisher\n")
            else:
                print("\nPrediction\n")
                live_instrument.computer_play(prediction=prediction)

        live_instrument.reset_sequence()
        if not gui.is_running:
            break

def vae_endless(gui):
    live_instrument = gui.live_instrument
    model = gui.model
    device = gui.device
    dials = gui.dials
    print("\nUser input\n")
    # reset live input clock and prerecorded sequences
    live_instrument.reset_sequence()
    live_instrument.reset_clock()
    while True:
        status_played_notes = live_instrument.clock()
        if status_played_notes:
            sequence = live_instrument.parse_to_matrix()
            live_instrument.reset_sequence()
            break
        if not gui.is_running:
            break

    while True:
        # send live recorded sequence through model and get response
        with torch.no_grad():
            # prepare sample for input
            sample = np.array(np.split(sequence, live_instrument.bars))
            sample = cutOctaves(sample)
            sample = torch.from_numpy(sample).float().to(device)
            sample = torch.unsqueeze(sample,1)

            # encode
            mu, logvar = model.encoder(sample)

            # reparameterize with variance
            dial_vals = []
            for dial in dials:
                dial_vals.append(dial.value())
            dial_tensor = torch.FloatTensor(dial_vals)/100.
            print(dial_tensor)
            new = mu + (dial_tensor * 0.5 * logvar.exp())
            pred = model.decoder(new).squeeze(1)

            # for more than 1 sequence
            prediction = pred[0]
            if pred.size(0) > 1:
                for p in pred[1:]:
                    prediction = torch.cat((prediction, p), dim=0)

            # back to cpu and normalize
            prediction = prediction.cpu().numpy()
            prediction /= np.abs(np.max(prediction))

            # check midi activations to include rests
            prediction[prediction < (1 - gui.slider_temperature.value()/100.)] = 0
            prediction = debinarizeMidi(prediction, prediction=True)
            prediction = addCuttedOctaves(prediction)
            smoother = NoteSmoother(prediction, threshold=2)
            prediction = smoother.smooth()

            # play predicted sequence note by note
            print("\nPrediction\n")
            live_instrument.computer_play(prediction=prediction)

        live_instrument.reset_sequence()
        sequence = prediction
        if not gui.is_running:
            break
