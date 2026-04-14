import torch
import torch.nn as nn

# Define a simple RNN manually
class SimpleRNN(nn.Module):
    def __init__(self, embedding_input_size: int, embedding_output_size: int, hidden_size: int, output_size: int):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_input_size = embedding_input_size
        
        self.embedding = nn.Embedding(embedding_input_size, embedding_output_size)
        self.rnn = nn.RNN(
            embedding_output_size, hidden_size, num_layers=2, nonlinearity="relu"
        )
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
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
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.hidden_size)