from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor
from rnn import SimpleRNN
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score



LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-7
EPOCHS = 20
LOG_INTERVAL = 1

def get_device():
    """Get device (cuda/cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, device=None, epochs=100, lr=0.1, weight_decay=0.01):
    if device is None:
        device = get_device()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        total = 0
        epoch_loss = 0.0
        model.train()
        for (X_batch, y_batch) in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total += len(y_batch)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total = 0
        total_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total += len(y_batch)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")
    
    return train_losses, val_losses


def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device=None):
    """Evaluate the model and return metrics."""
    if device is None:
        device = get_device()
    
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_preds.append(predictions.cpu())
            all_targets.append(y_batch.cpu())
    
    all_preds = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    
    accuracy = (all_preds == all_targets).sum() / len(all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'f1_macro': f1
    }
    
    return metrics

if __name__ == "__main__":
    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = KaggleSpamDataset("./data/emails.csv", 
        transform=torch.nn.Sequential(TokenizerTransform(tokenizer), ToTensor())
                                )
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = SimpleRNN(len(tokenizer) + 1, 256, 256, 2)
    
    print((model))
    
    model = model.to(device)
    
    print("\nTraining model...")
    train(model, train_loader, val_loader, device=device, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    print("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)
    
    print("\nMetrics:")
    print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}, Accuracy: {100*train_metrics['accuracy']:.2f}%")
    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}, Accuracy: {100*val_metrics['accuracy']:.2f}%")