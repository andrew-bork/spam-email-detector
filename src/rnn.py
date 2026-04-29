import torch
import torch.nn as nn

from tqdm import tqdm

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

    def train(self, train_loader, val_loader, device=None, epochs=100, lr=0.1, weight_decay=0.01):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            total = 0
            epoch_loss = 0.0
            self.train()
            for (X_batch, y_batch) in tqdm(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += len(y_batch)
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.eval()
            total = 0
            total_correct = 0
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = self(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    predictions = torch.argmax(outputs, dim=1)
                    total_correct += (predictions == y_batch).sum().item()
                    total += len(y_batch)
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")
        
        return train_losses, val_losses
        
