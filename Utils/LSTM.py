import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMCells(nn.Module):
    def __init__(self, input_fea, hidden_unit, recurrent_dropout = None):
        super(LSTMCells, self).__init__()
        self.hidden_dim = hidden_unit
        self.input_fea  = input_fea

        self.Linear1 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear2 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear3 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)
        self.Linear4 = nn.Linear(input_fea + hidden_unit, hidden_unit, bias=True)

        self.Linear5 = nn.Linear(hidden_unit, hidden_unit, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x, ct, ht): # [batch, Num_Cells, Depth] --> [batch, 1, Depth] 
        Input1 = torch.cat((ht, x), -1).float().to(device)

        sigma1 = self.sigmoid(self.Linear1(Input1))
        sigma2 = self.sigmoid(self.Linear2(Input1))
        sigma3 = self.sigmoid(self.Linear3(Input1))

        mul1 = sigma1 * ct
        mul2 = sigma2 * self.tanh(self.Linear4(Input1))

        c_next = mul1 + mul2

        # h_next = self.tanh(self.Linear5(ct)) * sigma3
        h_next = self.tanh(ct) * sigma3

        return c_next, h_next

        if self.return_sequence == False : return h1
        if self.return_sequence == True  : return Seq