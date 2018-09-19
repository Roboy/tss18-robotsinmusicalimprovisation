import torch
from torch import nn, optim
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, batch_size=7, lstm_layers=2, hidden_size=400, seq_length=7, input_size=100):
        super(LSTM, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.seq_length = seq_length
        self.input_size = input_size
        
        ###LSTM###########
        self.lstm = nn.LSTM(input_size=100, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers, batch_first=True, dropout=0)
        ##################
        
        ###LSTMCells######
        self.lstmC1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size, bias=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.lstmC2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.lstmC3 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True)
        #################
        
        self.fc = nn.Linear(self.hidden_size,self.input_size)
    
    def hiddenInitLSTM(self):
        hiddenState = torch.zeros(self.lstm_layers,
                                  int(self.batch_size/self.seq_length), 
                                  self.hidden_size).double().to(device)
        cellState = torch.zeros(self.lstm_layers,
                                int(self.batch_size/self.seq_length), 
                                self.hidden_size).double().to(device)
        return hiddenState, cellState
    
    def hiddenInitLSTMCell(self):
        hiddenState = torch.zeros(self.batch_size,
                                    self.hidden_size, 
                                    dtype=torch.double).to(device)
        cellState = torch.zeros(self.batch_size,
                                    self.hidden_size, 
                                    dtype=torch.double).to(device)
        hS2 = torch.zeros(self.batch_size,
                                    self.hidden_size, 
                                    dtype=torch.double).to(device)
        cS2 = torch.zeros(self.batch_size, self.hidden_size, 
                                    dtype=torch.double).to(device)
        hS3 = torch.zeros(self.batch_size, self.hidden_size, 
                                    dtype=torch.double).to(device)
        cS3 = torch.zeros(self.batch_size, self.hidden_size, 
                                    dtype=torch.double).to(device)
        
        return (hiddenState,cellState),(hS2,cS2),(hS3, cS3)

    def forward(self, embed, future = 0):#, lenghts):
 
        (h_t, c_t),(h2_t, c2_t),(h3_t,c3_t)= self.hiddenInitLSTMCell()
    
        #embedChunks = torch.chunk(embed, self.seq_length, dim=0)
        #print(embed.size())
        outputs = []
        for i in range(0,self.seq_length-1):
            h_t, c_t = self.lstmC1(embed[:,i,:],(h_t, c_t))
            h2_t, c2_t = self.lstmC2(h_t,(h2_t, c2_t))
            h3_t, c3_t = self.lstmC3(h2_t,(h3_t, c3_t))
            output = self.fc(h3_t)
            
        
        for i in range(future):
            h_t, c_t = self.lstmC1(output,(h_t, c_t))
            h2_t, c2_t = self.lstmC2(h_t, (h2_t, c2_t))
            #h3_t, c3_t = self.lstmC3(h2_t,(h3_t, c3_t))
            output = self.fc(h2_t)
            #outputs += [output]
            
        
        #print(outputs.shape)
        #if(self.training):
        #    print('embedding');print(embedChunks[0,-1:,:3])
        #    print('lstmOut');print(lstmOut[0,:1,:3])
        
        return embed, output