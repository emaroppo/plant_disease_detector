import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import json
from classes.models.PlantDiseaseCNN import PlantDiseaseCNN
from classes.PlantDiseaseDataset import PlantDiseaseDataset

root_dir = "data/PlantVillage/segmented"

with open("data/species_to_idx.json", "r") as f:
    species_to_idx = json.load(f)
with open("data/disease_to_idx.json", "r") as f:
    disease_to_idx = json.load(f)


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

dataset = PlantDiseaseDataset(
    root_dir, species_to_idx, disease_to_idx, transform=transform
)

model = PlantDiseaseCNN(
    num_species=len(species_to_idx.keys()), num_diseases=len(disease_to_idx.keys())
)


def train_model(model, dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model.train_model(
        train_dataloader,
        val_dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        num_epochs=10,  # starts overfitting early maybe prune or l
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # Save the model
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train_model(model, dataset)
    print("Training complete and model saved as model.pth")
