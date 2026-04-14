import torch
import torch.nn as nn

# Define a simple RNN manually
class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.internal_model = nn.Sequential(
            nn.Linear(input_size + output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        
        # Weight matrices
        self.W_ih = nn.Linear(input_size, hidden_size)    # Input to hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size)   # Hidden to hidden
        self.W_hy = nn.Linear(hidden_size, output_size)   # Hidden to output
        
    def forward(self, x, h_prev):
        """
        x: input of shape (batch_size, input_size)
        h_prev: previous hidden state of shape (batch_size, hidden_size)
        """
        # Compute new hidden state
        h_t = torch.tanh(self.W_ih(x) + self.W_hh(h_prev))
        
        # Compute output
        y_t = self.W_hy(h_t)
        
        return y_t, h_t
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.hidden_size)