import matplotlib
# matplotlib.use('Agg')
import numpy as np
import glob
import pypianoroll as ppr
import pretty_midi
import time
import music21
import os
import sys
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from utils.utilsPreprocessing import *
from loadModel import loadModel, loadStateDict
from tensorboardX import SummaryWriter
#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=50000)


class VAE(nn.Module):
    def __init__(self, embedding_size=100):
        super(VAE, self).__init__()

        self.embedding_size = embedding_size

        # ENCODER
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

        self.encode21 = nn.Sequential(
            nn.Linear(2400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Linear(400,200),
            nn.BatchNorm1d(200),
            nn.ELU(),
            nn.Linear(200,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ELU()
        )

        self.encode22 = nn.Sequential(
            nn.Linear(2400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Linear(400,200),
            nn.BatchNorm1d(200),
            nn.ELU(),
            nn.Linear(200,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ELU()
        )

        # DECODER
        self.decode1 = nn.Sequential(
            nn.Linear(self.embedding_size, 200),
            nn.BatchNorm1d(200),
            nn.ELU(),
            nn.Linear(200,400),
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

    def encoder(self, hEnc):
        #print("ENOCDER")
        hEnc = self.encode1(hEnc)
        hEnc = torch.squeeze(hEnc,3).view(-1,800*3)
        hEnc1 = self.encode21(hEnc)
        hEnc2 = self.encode22(hEnc)
        return hEnc1, hEnc2

    def decoder(self, z):
        #print("DECODER")
        hDec = self.decode1(z)
        hDec = hDec.view(hDec.size(0),800,-1).unsqueeze(2)
        hDec = self.decode2(hDec)
        return hDec

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.2*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            #print("no change")
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):

    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    #beta for disentanglement
    beta = 1e0

    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
    """
    batch = x.size(0)
    cosSim = torch.sum(cos(x.view(batch,-1),recon_x.view(batch,-1)))
    cosSim = batch-cosSim

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= mu.size(0) * mu.size(1)
    #print("loss values: cossim=",cosSim,"KLD=",KLD)
    return cosSim + (beta * KLD), cosSim, KLD


def train(epoch):
    model.train()
    trainLoss = 0
    cos_sims = 0
    klds = 0
    loss_divider = len(train_loader.dataset)-(len(train_loader.dataset)%batch_size)
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device)
        optimizer.zero_grad()
        reconBatch, mu, logvar = model(data)
        loss, cos_sim, kld = loss_function(reconBatch, data, mu, logvar)
        loss.backward()
        trainLoss += loss.item()
        cos_sims += cos_sim
        klds += kld
        optimizer.step()

        weights = []
        for i, f in enumerate(model.parameters()):
            weights.append(f.cpu().data.numpy())

        if(batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average Loss: {:.4f}'.format(
          epoch, trainLoss / loss_divider))

    return trainLoss / loss_divider, cos_sims.item() / loss_divider, \
                            klds.item() / loss_divider, weights, mu


def test(epoch):
    model.eval()
    test_loss = 0
    cos_sim = 0
    kld = 0
    loss_divider = len(test_loader.dataset)-(len(test_loader.dataset)%batch_size)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            reconBatch, mu, logvar = model(data)

            temp_test_loss, cos_sim_temp, kld_temp = loss_function(reconBatch, data, mu, logvar)
            test_loss += temp_test_loss.item()
            cos_sim += cos_sim_temp.item()
            kld += kld_temp.item()

    test_loss /= loss_divider

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, cos_sim / loss_divider, kld / loss_divider


if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='VAE settings')
    parser.add_argument("--file_path", default=None, help='Path to your MIDI files.')
    parser.add_argument("--checkpoint", default=None, help='Path to last checkpoint. \
    If you trained the checkpointed model on multiple GPUs use the --is_dataParallel flag. Default: None', type=str)
    parser.add_argument("--is_dataParallel", default=False, help='Option to allow loading models trained with multiple GPUs. Default: False', action="store_true")
    args = parser.parse_args()

    if not args.file_path:
        print("You have to set the path to your files from terminal with --file_path flag.")
        sys.exit()

    ################################################################################################
    ################################################################################################
    ################################################################################################
    # Hyperparameters
    epochs = 50                     # number of epochs you want to train for
    learning_rate = 1e-3            # starting learning rate
    learning_rate_decay = 0.9       # learning rate_decay per epoch
    lr_decay_step = 1               # step size of learning rate decay
    batch_size = 2000               # batch size of autoencoder
    log_interval = 25               # Log/show loss per batch
    embedding_size = 100            # size of latent vector
    beat_resolution = 24            # how many ticks per quarter note: 24 to process 1 bar at a time 12 for 2 bars
    seq_length = 96                 # how long is one sequence
    model_name = 'yamahapctpby60_new_arch_dataparrallel_1bar'
                                    # name for checkpoints / tensorboard
    ################################################################################################
    ################################################################################################
    ################################################################################################

    save_path = 'checkpoints/' + model_name
    writer = SummaryWriter(log_dir=('vae_plots/' + model_name))
    writer.add_text("learning_rate", str(learning_rate))
    writer.add_text("learning_rate_decay", str(learning_rate_decay))
    writer.add_text("learning_rate_decay_step", str(lr_decay_step))
    writer.add_text("batch_size", str(batch_size))
    writer.add_text("embedding_size", str(embedding_size))
    writer.add_text("beat_resolution", str(beat_resolution))
    writer.add_text("model_name", model_name)



    #create dataset
    if beat_resolution == 12:
        bars = 2
    else:
        bars = 1

    # check if train and test split already exists
    if os.path.isdir(args.file_path + 'train/') and os.path.isdir(args.file_path + 'test/'):
        print("train/ and test/ folder exist!")
        train_dataset = createDatasetAE(args.file_path + 'train/',
                                  beat_res = beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=True)

        test_dataset = createDatasetAE(args.file_path + 'test/',
                                  beat_res=beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    else:
        print("Only one folder with all files exist, using {}".format(args.file_path))
        dataset = createDatasetAE(args.file_path,
                                  beat_res=beat_resolution,
                                  bars=bars,
                                  seq_length = seq_length,
                                  binarize=True)
        train_size = int(np.floor(0.95 * len(dataset)))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("The training set contains {} sequences".format(len(train_dataset)))
    print("The test set contains {} sequences".format(len(test_dataset)))

    # IF YOU HAVE A BIG RAM YOU CAN SAVE THE WHOLE DATASET AS NPZ AND RUN IT FROM THERE
    """
    data = np.load('../WikifoniaPartlyNoTranspose.npz')
    midiDatasetTrain = data['train']
    midiDatasetTest = data['test']
    data.close()

    print(midiDatasetTrain.shape)
    print(midiDatasetTest.shape)

    train_loader = torch.utils.data.DataLoader(midiDatasetTrain, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(midiDatasetTest, batch_size=batch_size, shuffle=True, drop_last=True)
    """

    fullPitch = 128
    reducedPitch = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(embedding_size=embedding_size)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)
    writer.add_text("pytorch model", str(model))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load Checkpoint
    if args.checkpoint:
        print("Trying to load checkpoint...")
        if args.is_dataParallel:
            model = loadStateDict(model, args.checkpoint)
        else:
            model = loadModel(model, args.checkpoint, dataParallelModel=False)


    best_test_loss = np.inf
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=learning_rate_decay)
    for epoch in range(1, epochs + 1):
        scheduler.step()
        #training with plots
        train_loss, cos_sim_train, kld_train, weights, embedding = train(epoch)
        writer.add_scalar('loss/train_loss_epoch', train_loss, epoch)
        writer.add_scalar('loss/train_reconstruction_loss_epoch', cos_sim_train, epoch)
        writer.add_scalar('loss/train_kld_epoch', kld_train, epoch)
        for i, weight in enumerate(weights):
            writer.add_histogram(('weights/weight{}'.format(i)), weight, global_step=epoch)

        writer.add_histogram('embedding', embedding[0], bins='auto', global_step=epoch)


        #test
        test_loss, cos_sim_test, kld_test = test(epoch)
        writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
        writer.add_scalar('loss/test_reconstruction_loss_epoch', cos_sim_test, epoch)
        writer.add_scalar('loss/test_kld_epoch', kld_test, epoch)

        #save if model better than before
        if(test_loss < best_test_loss):
            best_test_loss = test_loss
            torch.save(model.state_dict(),(save_path + '.pth'))

    writer.close()
