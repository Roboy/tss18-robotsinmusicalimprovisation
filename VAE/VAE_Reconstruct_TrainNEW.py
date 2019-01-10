#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import numpy as np
import glob
import pypianoroll as ppr
import pretty_midi
import time
import music21
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from utils.utilsPreprocessing import *

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=50000)


##################################
#HYPERPARAMS
##################################
epochs = 10
learning_rate = 1e-4
batch_size = 200
log_interval = 100 #Log/show loss per batch
embedding_size = 100
beat_resolution = 24
seq_length = 96


# In[ ]:


#create dataset
from torch.utils.data.sampler import SubsetRandomSampler

path_to_files = "/media/EXTHD/niciData/DATASETS/AllSequences/"

dataset = createDatasetAE((path_to_files + "*.npy"), 
                           beat_res = beat_resolution, 
                           seq_length = seq_length,
                           binarize=True)
print("Dataset contains {} sequences".format(len(dataset)))
    
train_size = int(np.floor(0.9 * len(dataset)))
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


print("The training set contains {} sequences".format(len(train_dataset)))
print("The validation set contains {} sequences".format(len(valid_dataset)))


# In[ ]:


#load dataset from npz
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
print('')


# In[ ]:


fullPitch = 128; reducedPitch = 60


# # VAE model

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


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
            
        self.encode2 = nn.Sequential(
            nn.Linear(2400,800),
            nn.BatchNorm1d(800),
            nn.ELU(),
            nn.Linear(800,400),
            nn.BatchNorm1d(400),
            nn.ELU()
            #nn.Linear(400,100),
            #nn.BatchNorm1d(100),
            #nn.ELU()
        )
        self.encode31 = nn.Sequential(
            nn.Linear(400,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ELU()
        )
        self.encode32 = nn.Sequential(
            nn.Linear(400,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ELU()
        )

        # DECODER
        self.decode1 = nn.Sequential(
            nn.Linear(self.embedding_size,400),
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
        hEnc = self.encode2(hEnc)
        hEnc1 = self.encode31(hEnc)
        hEnc2 = self.encode32(hEnc)
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
    
model = VAE(embedding_size=embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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
        ###DENOISING AUTOENCODER
        #data = data.float().to(device)
        #noise = torch.bernoulli((torch.rand_like(data))).to(device)
        #print(noise[0])
        #noisyData = data+noise
        #reconBatch, mu = model(noisyData)
        
        ###NORMAL AUTOENCODER
        data = data.float().to(device)
        optimizer.zero_grad()
        reconBatch, mu, logvar = model(data)

        loss, cos_sim, kld = loss_function(reconBatch, data, mu, logvar)
        loss.backward()
        trainLoss += loss.item()
        cos_sims += cos_sim
        klds += kld
        optimizer.step()
        if(batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average Loss: {:.4f}'.format(
          epoch, trainLoss / loss_divider))
    
    return trainLoss / loss_divider, cos_sims / loss_divider, klds / loss_divider

def validate(epoch):
    model.eval()
    valid_loss = 0
    cos_sim = 0
    kld = 0
    loss_divider = len(valid_loader.dataset)-(len(valid_loader.dataset)%batch_size)
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            ###DENOISING AUTOENCODER
            #data = data.float().to(device)
            #noise = torch.bernoulli((torch.rand_like(data))).to(device)
            #noisyData = data+noise
            #reconBatch, mu = model(noisyData)
            
            ###NORMAL AUTOENCODER
            data = data.float().to(device)
            reconBatch, mu, logvar = model(data)
            
            temp_valid_loss, cos_sim_temp, kld_temp = loss_function(reconBatch, data, mu, logvar)
            valid_loss += temp_valid_loss.item()
            cos_sim += cos_sim_temp.item()
            kld += kld_temp.item()
            #if(i==10):
            #    break
    valid_loss /= loss_divider

    print('====> Test set loss: {:.4f}'.format(valid_loss))
    return valid_loss, cos_sim, kld


# # Load Model

# In[ ]:


#from loadModel import loadModel
#pathToModel = '../models/YamahaPC2002_VAE_Reconstruct_NoTW_20Epochs.model'

#model = loadModel(model,pathToModel, dataParallelModel=False)


# # Train

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
save_path = '/media/EXTHD/niciData/models/VAE_YamahaPC2002_{}LR_{}SeqLength'.format(str(learning_rate), seq_length)

train_losses = []
valid_losses = []
cos_sims_train = []
klds_train = []
cos_sims_test = []
klds_test = []
best_valid_loss = 999.
for epoch in range(1, epochs + 1):
    #training with plots
    train_loss, cos_sim_train, kld_train = train(epoch)
    train_losses.append(train_loss)
    cos_sims_train.append(cos_sim_train)
    klds_train.append(kld_train)
    
    #validate with plots
    current_valid_loss, cos_sim_test, kld_test = validate(epoch)
    valid_losses.append(current_valid_loss)
    cos_sims_test.append(cos_sim_test)
    klds_test.append(kld_test)
    
    #save if model better than before
    if(current_valid_loss < best_valid_loss):
        best_valid_loss = current_valid_loss
        torch.save(model.state_dict(),(save_path + '.pth'))
        
    #plot current results
    plt.plot(train_losses, color='red', label='Train Loss')
    plt.plot(valid_losses, color='orange', label='Test Loss')
    plt.plot(cos_sims_train, color='blue', label='Cosine Similarity Train')
    plt.plot(klds_train, color='black', label='KL Divergence Train')
    plt.plot(cos_sims_test, color='green', label='Cosine Similarity Test')
    plt.plot(klds_test, color='cyan', label='KL Divergence Test')
    plt.legend(loc='upper right')
    plt.show()
    print('')
        

plt.plot(train_losses, color='red', label='Train Loss')
plt.plot(valid_losses, color='orange', label='Test Loss')
plt.plot(cos_sims_train, color='blue', label='Cosine Similarity Train')
plt.plot(klds_train, color='black', label='KL Divergence Train')
plt.plot(cos_sims_test, color='green', label='Cosine Similarity Test')
plt.plot(klds_test, color='cyan', label='KL Divergence Test')
plt.legend(loc='upper right')
plt.savefig(save_path + '.png')
plt.show()
print('')


# In[ ]:


#torch.save(model,'/media/EXTHD/niciData/models/YamahaPC2002_DenoisingCVAE_Reconstruct_NoTW_5Epochs.model')


# # Generate

# In[ ]:


#np.set_printoptions(precision=4, suppress=True, threshold=np.inf)


# In[ ]:


"""
if(model.train()):
    model.eval()
    
###PLAY WHOLE SONG IN BARS
with torch.no_grad():
    sampleNp1 = getSlicedPianorollMatrixNp("../WikifoniaDatabase/test/Hermann-Lohr,-D.-Eardley-Wilmot---Little-Grey-Home-In-The-West.mid")
    sampleNp1 = deleteZeroMatrices(sampleNp1)
    
    for i,sampleNp in enumerate(sampleNp1[:8]):
        sampleNp = sampleNp[:,36:-32]
        sample = torch.from_numpy(sampleNp).float()
        embed, logvar = model.encoder(sample.reshape(1,1,seq_length,reducedPitch).to(device))
        ###RECONSTRUCTION#########
        pred = model.decoder(embed)
        ##########################
        ###RANDOM RECONSTRUCTION##
        std = torch.exp(0.2*logvar)
        eps = torch.randn_like(std)
        randomRecon = model.decoder(eps.mul(std).add_(embed))
        ##########################

        reconstruction = pred.squeeze(0).squeeze(0).cpu().numpy()
        randomRecon = randomRecon.squeeze(0).squeeze(0).cpu().numpy()

        #NORMALIZE PREDICTIONS
        reconstruction /= np.abs(np.max(reconstruction))
        randomRecon /= np.abs(np.max(randomRecon))

        #CHECK MIDI ACTIVATIONS IN PREDICTION TO INCLUDE RESTS
        reconstruction[reconstruction < 0.2] = 0
        randomRecon[randomRecon < 0.3] = 0

        samplePlay = debinarizeMidi(sampleNp, prediction=False)
        samplePlay = addCuttedOctaves(samplePlay)
        reconstruction = debinarizeMidi(reconstruction, prediction=True)
        reconstruction = addCuttedOctaves(reconstruction)
        randomRecon = debinarizeMidi(randomRecon, prediction=True)
        randomRecon = addCuttedOctaves(randomRecon)
        if(i==0):
            sampleOut = samplePlay
            reconOut = reconstruction
            randomReconOut = randomRecon
        else:
            sampleOut = np.concatenate((sampleOut,samplePlay), axis=0)
            reconOut = np.concatenate((reconOut,reconstruction), axis=0)
            randomReconOut = np.concatenate((randomReconOut, randomRecon),axis=0)


    print("INPUT")
    pianorollMatrixToTempMidi(sampleOut, show=True,showPlayer=True,autoplay=False,
                             path='../temp/inputTemp.mid')
    print("RECONSTRUCTION")
    pianorollMatrixToTempMidi(reconOut, show=True,showPlayer=True,autoplay=False,
                             path='../temp/reconTemp.mid')
    #print("Reconstruction with Noise")
    #pianorollMatrixToTempMidi(randomReconOut, show=True,showPlayer=True,autoplay=False,
    #                         path='../temp/noiseReconTemp.mid')  
    print("\n\n")
            
"""
print('')


# # Morph from one sequence to another in latent space
# 
# Take 2 embedded sequences and slowly morph it to the other one based on this formula: $c_\alpha=(1-\alpha)*z_1 + \alpha*z_2$ , where $z_1$ corresponds to the embedding of sample 1 and $z_2$ to sample 2.
# (Source of formula: https://arxiv.org/pdf/1803.05428.pdf)

# In[ ]:


"""
with torch.no_grad():
    #get 2 different unseeen sequences and choose random sequence
    sample1 = getSlicedPianorollMatrixNp('../WikifoniaDatabase/train/1952,-Jerry-Lieber-&-Mike-Stoller---Kansas-City.mid')
    sample1 = sample1[6,:,36:-32]
    sample2 = getSlicedPianorollMatrixNp('../WikifoniaDatabase/test/Hoagy-Carmichael---Blue-Orchids.mid')
    sample2 = sample2[7,:,36:-32]
    print('sample1 shape:', sample1.shape)
    print('sample2 shape: ',sample2.shape)

    #prepare for input
    sample1 = torch.from_numpy(sample1.reshape(1,1,96,60)).float().to(device)
    sample2 = torch.from_numpy(sample2.reshape(1,1,96,60)).float().to(device)
    
    #run through encoder
    embed1, _ = model.encoder(sample1)
    embed2, _ = model.encoder(sample2)
    
    for a in range(0,11):
        alpha = a/10
        print("alpha is ",alpha)
        c = (1-alpha) * embed1 + alpha * embed2
        
        #decode current
        recon = model.decoder(c)
        recon = recon.squeeze(0).squeeze(0).cpu().numpy()
        recon /= np.max(np.abs(recon))
        recon[recon < 0.2] = 0
        recon = debinarizeMidi(recon, prediction=True)
        recon = addCuttedOctaves(recon)
        pianorollMatrixToTempMidi(recon, show=True,showPlayer=False,autoplay=True,
                             path='../temp/temp.mid')
        
"""       
print('')        


# In[ ]:




