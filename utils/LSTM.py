import torch
from torch import nn, optim
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, batch_size=7, lstm_layers=2, hidden_size=32, seq_length=7, input_size=100):
        super(LSTM, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.seq_length = seq_length
        self.input_size = input_size
        
        ###LSTM###########
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers, batch_first=True, dropout=0.3)
        ##################
        
        ###LINEAR LAYERS###
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size,self.input_size)
        ###################    
    
    def reorder_batch(self, embedded_seqs):
        return torch.stack(torch.chunk(embedded_seqs, int(self.batch_size/self.seq_length)))
        
    
    def hidden_init(self):
        return torch.zeros(self.lstm_layers, int(self.batch_size/self.seq_length), 
                           self.hidden_size, dtype=torch.double).to(device)

    def forward(self, embed, future = 0):
        print(self)
        print(self.batch_size, self.seq_length, self.hidden_size)
        h_t0 = self.hidden_init()
        c_t0 = self.hidden_init()
        
        lstm_input = self.i2h(embed)
        output, (h_t1, c_t1) = self.lstm(lstm_input, (h_t0, c_t0))
        output = self.h2o(output[:,:,:])
        
        return embed, output[:,-1,:]

    