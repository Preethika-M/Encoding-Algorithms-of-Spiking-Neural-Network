import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the training function
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode
    loss_history = []  # List to store loss per epoch

    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass
            spk_rec, mem_rec = model(signals)

            # Compute the loss across time steps
            loss_val = torch.zeros((1), dtype=torch.float32, device=device)
            for step in range(model.num_steps):
                loss_val += criterion(mem_rec[step], labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return loss_history

# Hyperparameters and data loading
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# DataLoader for training dataset
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model and plot loss history
loss_history = train(model, train_loader, criterion, optimizer, num_epochs, device)

# Plot training loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
