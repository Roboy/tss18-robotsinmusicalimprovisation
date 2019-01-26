import torch
import numpy as np
from utils.utils import (cutOctaves, debinarizeMidi, addCuttedOctaves)


def vae_interact(gui):
    live_instrument = gui.live_instrument
    model = gui.model
    device = gui.device
    dials = gui.dials
    while True:# reset live input clock]
        print("\nUser input\n")
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
            for dial in dials:
                dial_vals.append(dial.value())
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
            prediction[prediction < (1 - gui.slider_temperature.value()/100.)] = 0
            prediction = debinarizeMidi(prediction, prediction=True)
            prediction = addCuttedOctaves(prediction)

            # play predicted sequence note by note
            print("\nPrediction\n")
            live_instrument.computer_play(prediction=prediction)

        live_instrument.reset_sequence()
        if not gui.is_running:
            break
