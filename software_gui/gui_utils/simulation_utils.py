import torch
import numpy as np
from utils.utils import (cutOctaves, debinarizeMidi, addCuttedOctaves)
from utils.NoteSmoother import NoteSmoother
import roslib
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Int32
import mido

def numpy_publisher(pub, prediction):
    r = rospy.Rate(10) # 10hz
    msg = Float32MultiArray()
    msg.data = prediction
    pub.publish(msg)
    r.sleep()

def vae_interact(gui):
    live_instrument = gui.live_instrument
    device = gui.device
    model = gui.model.to(device)
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
            dial_tensor = (torch.FloatTensor(dial_vals)/100.).to(device)
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

            # sent to robot
            if gui.chx_simulate_robot.isChecked():
                print("\nPublisher\n")
                note_msg = Int32MultiArray()
                live_instrument.human = False
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
                                    mido_msg = mido.Message('note_on', note=note, velocity=current_vel)
                                    note_msg.data = mido_msg.bytes()
                                    gui.midi_publisher.publish(note_msg)
                                    played_notes.append(note)
                        else:
                            for note in played_notes:
                                # self.out_port.send(mido.Message('note_off',
                                #                             note=note))#, velocity=100))
                                played_notes.pop(0)

                        if old_midi_on.any():
                            for note in old_midi_on[0]:
                                if note not in midi_on:
                                    # self.out_port.send(mido.Message('note_off', note=note))
                                    continue
                        old_midi_on = midi_on

                    if done:
                        live_instrument.human = True
                        live_instrument.reset_clock()
                        break
            # or play in software
            else:
                print("\nPrediction\n")
                live_instrument.computer_play(prediction=prediction)

        live_instrument.reset_sequence()
        if not gui.is_running:
            break

def vae_endless(gui):
    live_instrument = gui.live_instrument
    device = gui.device
    model = gui.model.to(device)
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
            dial_tensor = (torch.FloatTensor(dial_vals)/100.).to(device)
            # print(dial_tensor)
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

            # sent to robot
            if gui.chx_simulate_robot.isChecked():
                print("\nPublisher\n")
                note_msg = Int32MultiArray()
                live_instrument.human = False
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
                                    mido_msg = mido.Message('note_on', note=note, velocity=current_vel)
                                    note_msg.data = mido_msg.bytes()
                                    gui.midi_publisher.publish(note_msg)
                                    played_notes.append(note)
                        else:
                            for note in played_notes:
                                # self.out_port.send(mido.Message('note_off',
                                #                             note=note))#, velocity=100))
                                played_notes.pop(0)

                        if old_midi_on.any():
                            for note in old_midi_on[0]:
                                if note not in midi_on:
                                    # self.out_port.send(mido.Message('note_off', note=note))
                                    continue
                        old_midi_on = midi_on

                    if done:
                        live_instrument.human = True
                        live_instrument.reset_clock()
                        break
            # or play in software
            else:
                print("\nPrediction\n")
                live_instrument.computer_play(prediction=prediction)

        live_instrument.reset_sequence()
        sequence = prediction
        if not gui.is_running:
            break
