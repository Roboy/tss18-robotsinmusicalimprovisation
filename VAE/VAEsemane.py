import argparse
import sys
import numpy as np
import torch
import torch.utils.data
from utils.raspi_utils import (cutOctaves, debinarizeMidi, addCuttedOctaves)
from VAE.VAE_model import VAE
from loadModel import loadModel, loadStateDict
from utils.LiveInput import LiveParser
from mido import MidiFile, Message
np.set_printoptions(precision=1, threshold=np.inf)


def vae_main(live_instrument, model, args):
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
        sample = np.array(np.split(sequence, args.bars))

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
        prediction[prediction < (1 - args.temperature)] = 0
        prediction = debinarizeMidi(prediction, prediction=True)
        prediction = addCuttedOctaves(prediction)

        # play predicted sequence note by note
        print("\nPrediction\n")
        live_instrument.computer_play(prediction=prediction)


    live_instrument.reset_sequence()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAEsemane')
    parser.add_argument("--port", default=None,
                        help='MIDI controller input port. If you want to set \
                        it from bash. You can also set it later. \
                        Default = None.', type=str)
    parser.add_argument("--model_path",
                        help='Path to Autoencoder model. If you trained the AE \
                        on multiple GPUs use the --AE_model_dataParallel flag. \
                        Defaults to pretrained model.', type=str)
    parser.add_argument("--is_dataParallel", default=False, dest='is_dataParallel',
                        help='Option to allow loading models trained with \
                        multiple GPUs. Default: False', action="store_true")
    parser.add_argument("--temperature", default=0.7,
                        help='Temperature of improvisation. Default: 0.7',
                        type=float)
    parser.add_argument("--bars", default=1,
                        help='Set the number of bars you want to record. \
                        Default: 1', type=int)
    parser.add_argument("--bpm", default=120,
                        help='Set the beats per minute (BPM). Default: 120',
                        type=int)
    parser.add_argument("--input_size", default=100,
                        help='If you trained an autoencoder with a different \
                        embedding size, you can change it here. Default: 100',
                        type=int)
    parser.set_defaults(is_dataParallel=False)
    args = parser.parse_args()

    if not args.model_path:
        print("Please set the model path to autoencoder model using \
            --model_path /path/to/model")
        sys.exit()

    # initialize live input instrument
    live_instrument = LiveParser(port=args.port, bars=args.bars,
                                bpm=args.bpm, ppq=24)
    live_instrument.open_inport(live_instrument.parse_notes)
    live_instrument.open_outport()

    #load model and corresponding weights
    model = VAE()
    try:
        model = loadStateDict(model, args.model_path)
    except:
        model = loadModel(model, args.model_path)

    # check for gpu support and send model to gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # make sure model is in eval mode
    if model.train():
        model.eval()

    while True:
        vae_main(live_instrument, model, args)
