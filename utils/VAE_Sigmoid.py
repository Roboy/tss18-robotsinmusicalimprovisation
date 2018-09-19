import torch
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, embedding_size=100):
        super(VAE, self).__init__()
        
        self.embedding_size = embedding_size
        
        ###ENCODER###
        self.encode1 = nn.Sequential(
            nn.Conv2d(1,100,(16,5),stride=(16,5),padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100,200,(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200,400,(2,2),stride=(1,2),padding=0),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Conv2d(400,800,(2,2),stride=(2,2),padding=0),
            nn.BatchNorm2d(800),
            nn.ReLU()
        )
            
        self.encode2 = nn.Sequential(
            nn.Linear(2400,800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800,400),
            nn.BatchNorm1d(400),
            nn.ReLU()
            #nn.Linear(400,100),
            #nn.BatchNorm1d(100),
            #nn.ELU()
        )
        self.encode31 = nn.Sequential(
            nn.Linear(400,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()
        )
        self.encode32 = nn.Sequential(
            nn.Linear(400,self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()
        )

        ###DECODER###
        self.decode1 = nn.Sequential(
            nn.Linear(self.embedding_size,400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400,800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800,2400),
            nn.BatchNorm1d(2400),
            nn.ReLU()
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(800,400,(2,2),stride=(2,2),padding=0),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.ConvTranspose2d(400,200,(2,2),stride=(1,2),padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.ConvTranspose2d(200,100,(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.ConvTranspose2d(100,1,(16,5),stride=(16,5),padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
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
        hDec = hDec.view(hDec.size()[0],800,-1).unsqueeze(2)
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