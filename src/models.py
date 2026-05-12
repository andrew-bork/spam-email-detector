import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, embedding_output_size: int = 512, hidden_size: int = 512, output_size: int = 2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.embedding = nn.Embedding(input_size, embedding_output_size)
        self.rnn = nn.RNN(
            embedding_output_size, hidden_size, num_layers=2, nonlinearity="relu"
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x: torch.Tensor):
        """
        x: input of shape (batch_size, input_size)
        h_prev: previous hidden state of shape (batch_size, hidden_size)
        """
        # Compute new hidden state
        x = self.embedding(x)
        h0 = torch.zeros(2, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.output_layer(out[:, -1, :])
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, 
            input_size: int,
            hidden_size: int = 256,
            output_size: int = 2  
        ):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )
    
    def forward(self, x: torch.Tensor):
        return self.model(x)

class LinearRegression(nn.Module):
    def __init__(self, 
            input_size: int,
            output_size: int = 1
        ):
        super(LinearRegression, self).__init__()
        self.model = nn.Linear(input_size, output_size)
        
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    
    
    
    
class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,           # vocab size (e.g. 30522 for bert-base-cased)
        embedding_output_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.3,
    ):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
 
        # Maps token ids → dense vectors
        self.embedding = nn.Embedding(input_size, embedding_output_size, padding_idx=0)
 
        # Stacked bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input/output: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
 
        self.dropout = nn.Dropout(dropout)
 
        # ×2 because bidirectional
        self.output_layer = nn.Linear(hidden_size * 2, output_size)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len)  — padded token-id tensor
        returns logits : (batch, output_size)
        """
        embedded = self.dropout(self.embedding(x))          # (B, T, E)
        output, (h_n, _) = self.lstm(embedded)              # output: (B, T, 2H)
 
        # Concatenate final forward and backward hidden states from the last layer
        forward_hidden  = h_n[-2]                           # (B, H)
        backward_hidden = h_n[-1]                           # (B, H)
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, 2H)
 
        return self.output_layer(self.dropout(combined))    # (B, output_size)
