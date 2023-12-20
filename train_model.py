from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):
    # Initialize batch size and number of epochs
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    # losses = {}
    loss_function = nn.BCEWithLogitsLoss()
    # min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    # losses.append(min_loss)

    training_losses = []
    testing_losses = []
    clip_value = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

    for epoch_i in range(no_epochs):
        model.train()
        running_loss = 0.0
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            optimizer.zero_grad()
            # print(sample['input'])
            # print(sample['label'])
            output = model(sample['input'])
            label = sample['label'].float()
            loss = loss_function(output, label.unsqueeze(1))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            # print(loss)
            running_loss += loss.item()
            
        average_training_loss = running_loss / len(data_loaders.train_loader)
        training_losses.append(average_training_loss)

        # Evaluate the model on the testing data
        testing_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        testing_losses.append(testing_loss)
        
        print(f'Epoch {epoch_i + 1}/{no_epochs} - Training Loss: {average_training_loss:.4f}, Testing Loss: {testing_loss:.4f}')
            
        torch.save(model.state_dict(), "saved/saved_model.pkl")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(testing_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    no_epochs = 3
    train_model(no_epochs)
