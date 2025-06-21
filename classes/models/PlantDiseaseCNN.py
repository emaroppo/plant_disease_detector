import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_species=14, num_diseases=21):
        super(PlantDiseaseCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 32 * 32 + num_species, 512)
        self.fc2 = nn.Linear(512, num_diseases)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, species):
        # Image feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the image
        x = x.view(-1, 64 * 32 * 32)
        x = self.dropout(x)

        # Concatenate the one-hot encoded species vector directly with the flattened image features
        x = torch.cat((x, species), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train_model(
        self,
        dataloader,
        validation_dataloader,
        loss_fn,
        optimizer,
        num_epochs=10,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.to(device)
        writer = SummaryWriter()
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch, (images, species, labels) in enumerate(dataloader):
                images = images.to(device)
                species = (
                    F.one_hot(species, num_classes=14).to(torch.float32).to(device)
                )
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self(images, species)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100 * correct_predictions / total_predictions

            writer.add_scalar("Training Loss", epoch_loss, epoch)
            writer.add_scalar("Training Accuracy", epoch_accuracy, epoch)
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )

            if validation_dataloader is not None:
                self.eval()
                val_loss, val_accuracy = self.validate(
                    validation_dataloader, loss_fn, device
                )
                writer.add_scalar("Validation Loss", val_loss, epoch)
                writer.add_scalar("Validation Accuracy", val_accuracy, epoch)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        writer.close()

    def validate(self, dataloader, loss_fn, device):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, species, labels in dataloader:
                images = images.to(device)
                species = (
                    F.one_hot(species, num_classes=14).to(torch.float32).to(device)
                )
                species = species.to(device)
                labels = labels.to(device)

                outputs = self(images, species)
                loss = loss_fn(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        average_loss = running_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions
        return average_loss, accuracy
