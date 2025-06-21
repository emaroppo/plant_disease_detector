from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, species_to_idx, disease_to_idx, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            species_to_idx (dict): A dictionary mapping species to numerical labels.
            disease_to_idx (dict): A dictionary mapping diseases to numerical labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.species_to_idx = species_to_idx
        self.disease_to_idx = disease_to_idx
        self.transform = transform
        self.samples = []

        for dir_name, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.endswith(".jpg"):
                    path = os.path.join(dir_name, file_name)
                    folder_name = os.path.basename(dir_name)
                    species, disease = folder_name.split("___")
                    species_idx = species_to_idx[species]
                    disease_idx = disease_to_idx[disease]
                    self.samples.append((path, species_idx, disease_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, species_idx, disease_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        species_idx = torch.tensor(species_idx, dtype=torch.long)
        disease_idx = torch.tensor(disease_idx, dtype=torch.long)

        return image, species_idx, disease_idx
