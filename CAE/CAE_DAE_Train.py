import sys
import os
import numpy as np
import glob
import pypianoroll as ppr
import time
import music21
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from utils.utils import *
import argparse
from tensorboardX import SummaryWriter
from utils.createDatasetAE import createDatasetAE, loadDatasets
#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=50000)

class CAE_DAE(nn.Module):
    def __init__(self):
        super(CAE_DAE, self).__init__()

        # Encoder
        self.encode1 = nn.Sequential(
            nn.Conv2d(1,100,(16,5),stride=(16,5),padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.Conv2d(100,200,(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.Conv2d(200,400,(2,2),stride=(1,2),padding=0),
            nn.BatchNorm2d(400),
            nn.ELU(),
            nn.Conv2d(400,800,(2,2),stride=(2,2),padding=0),
            nn.BatchNorm2d(800),
            nn.ELU()
        )

        self.encode2 = nn.Sequential(
            nn.Linear(2400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Linear(400,100),
            nn.BatchNorm1d(100),
            nn.ELU()
        )

        # Decoder
        self.decode1 = nn.Sequential(
            nn.Linear(100,400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Linear(400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,2400),
            nn.BatchNorm1d(2400),
            nn.ELU()
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(800,400,(2,2),stride=(2,2),padding=0),
            nn.BatchNorm2d(400),
            nn.ELU(),
            nn.ConvTranspose2d(400,200,(2,2),stride=(1,2),padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.ConvTranspose2d(200,100,(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.ConvTranspose2d(100,1,(16,5),stride=(16,5),padding=0),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def encoder(self, x):
        #print("ENOCDER")
        hEnc = self.encode1(x)
        hEnc = torch.squeeze(hEnc,3).view(-1,800*3)
        hEnc = self.encode2(hEnc)
        return hEnc

    def decoder(self, z):
        #print("DECODER")
        hDec = self.decode1(z)
        hDec = hDec.view(hDec.size()[0],800,-1).unsqueeze(2)
        hDec = self.decode2(hDec)
        return hDec

    def forward(self, x):
        mu = self.encoder(x)
        return self.decoder(mu), mu


def loss_function(x, recon_x):
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    cosSim = 0
    for datapoint, recon in zip(x, recon_x):
        cosTemp = cos(datapoint.view(1,-1),recon.view(1,-1))
        cosSim += cosTemp
    # import pdb; pdb.set_trace()
    return x.size()[0]-cosSim


def train(epoch, denoise=False):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device)
        optimizer.zero_grad()
        if denoise:
            noise = torch.bernoulli((torch.rand_like(data))).to(device)
            noisy_data = data + noise
            reconBatch, mu = model(noisy_data)
        else:
            reconBatch, mu = model(data)
        loss = loss_function(data, reconBatch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if(batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))
    return train_loss


def test(epoch, data_loader, denoise=False, validate=False):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.float().to(device)
            if denoise:
                noise = torch.bernoulli((torch.rand_like(data))).to(device)
                noisy_data = data + noise
                reconBatch, mu = model(noisy_data)
            else:
                reconBatch, mu = model(data)
            loss += loss_function(data, reconBatch).item()

    loss /= len(data_loader.dataset)
    if validate:
        print('====> Valid set loss: {:.4f}'.format(loss))
    else:
        print('====> Test set loss: {:.4f}'.format(loss))

    return loss

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='VAE settings')
    parser.add_argument("--file_path", default=None,
        help='Path to your MIDI files.')
    parser.add_argument("--validation_path", default=None,
        help='Path to your validation set. You should take this from a different dataset.')
    # parser.add_argument("--checkpoint", default=None, help='Path to last checkpoint. \
    #     If you trained the checkpointed model on multiple GPUs use the --is_dataParallel flag. \
    #     Default: None', type=str)
    # parser.add_argument("--is_dataParallel", default=False,
    #     help='Option to allow loading models trained with multiple GPUs. \
    #     Default: False', action="store_true")
    args = parser.parse_args()

    if not args.file_path:
        print("You have to set the path to your files from terminal with \
         --file_path flag.")
        sys.exit()

    ############################################################################
    ############################################################################
    ############################################################################
    # Hyperparameters
    epochs = 25                     # number of epochs you want to train for
    learning_rate = 1e-3            # starting learning rate
    learning_rate_decay = 0.1       # learning rate_decay per epoch
                                        # set None to turn off
    lr_decay_step = 1               # step size of learning rate decay
    batch_size = 100                # batch size of autoencoder
    log_interval = 50               # Log/show loss per batch
    beat_resolution = 24            # how many ticks per quarter note:
                                        # 24 to process 1 bar at a time 12 for 2
    seq_length = 96                 # how long is one sequence in MIDI ticks
    denoising_ae = True             # set this to true if you want to train a DAE
    model_name = 'DAE_dataset_name'
                                    # name for checkpoints / tensorboard
    ############################################################################
    ############################################################################
    ############################################################################

    # tensorboard
    save_path = 'checkpoints/' + model_name
    writer = SummaryWriter(log_dir=('cae_dae_plots/' + model_name))
    writer.add_text("learning_rate", str(learning_rate))
    writer.add_text("learning_rate_decay", str(learning_rate_decay))
    writer.add_text("learning_rate_decay_step", str(lr_decay_step))
    writer.add_text("batch_size", str(batch_size))
    writer.add_text("beat_resolution", str(beat_resolution))
    writer.add_text("model_name", model_name)

    # create datasets
    train_loader, test_loader, valid_loader = loadDatasets(args.file_path,
        args.validation_path, batch_size=batch_size, beat_resolution=24)

    print("The training set contains {} sequences".format(len(train_loader.dataset)))
    print("The test set contains {} sequences".format(len(test_loader.dataset)))
    if args.validation_path:
        print("The valdiation set contains {} sequences".format(len(valid_loader.dataset)))

    # instantiate model and if possbile make multi-GPU
    model = CAE_DAE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)
    writer.add_text("pytorch model", str(model))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """
    # load pretrained model here

    """

    best_valid_loss = np.inf
    if learning_rate_decay:
        print("Learning rate decay activated!")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=learning_rate_decay)
    for epoch in range(1, epochs + 1):
        if learning_rate_decay:
            scheduler.step()
        #training with plots
        train_loss = train(epoch, denoise=denoising_ae)
        writer.add_scalar('loss/train_loss_epoch', train_loss, epoch)
        #test
        test_loss = test(epoch, data_loader=test_loader, denoise=denoising_ae)
        writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
        #validate
        valid_loss = test(epoch, data_loader=valid_loader,
                            denoise=denoising_ae, validate=True)
        writer.add_scalar('loss/valid_loss_epoch', valid_loss, epoch)

        #save if model better than before
        if (valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),(save_path + '.pth'))

    writer.close()
