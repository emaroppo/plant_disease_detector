import streamlit as st
from PIL import Image
from torchvision import transforms
import json

with open("data/species_to_idx.json", "r") as f:
    species_to_idx = json.load(f)
with open("data/disease_to_idx.json", "r") as f:
    disease_to_idx = json.load(f)

# select model from dropdown
state_dict_path = st.selectbox(
    "Select a model",
    options=["model.pth", "model_v2.pth"],
    index=0,
)
# Load the model
import torch
from classes.models.PlantDiseaseCNN import PlantDiseaseCNN


@st.cache_resource
def load_model(state_dict_path):
    model = PlantDiseaseCNN(num_species=14, num_diseases=21)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model(state_dict_path)


# 3 Tabs
# Tab 1 - Predict

# submit a photo and a species, generate prediction

image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
species = st.selectbox(
    "Select a species",
    options=species_to_idx.keys(),
    index=0,
)

if st.button("Predict"):
    if image is not None and species:
        # Load and preprocess the image
        img = Image.open(image).convert("RGB")
        img = img.resize((256, 256))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)

        # Convert species to one-hot encoding
        species_idx = {
            "Apple": 0,
            "Blueberry": 1,
            "Cherry": 2,
            "Corn": 3,
            "Grape": 4,
            "Peach": 5,
            "Pepper": 6,
            "Potato": 7,
            "Raspberry": 8,
            "Soybean": 9,
            "Squash": 10,
            "Strawberry": 11,
            "Tomato": 12,
        }[species]
        species_one_hot = (
            torch.nn.functional.one_hot(torch.tensor(species_idx), num_classes=14)
            .float()
            .unsqueeze(0)
        )

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor, species_one_hot)
            _, predicted_disease = torch.max(output, 1)

        # Map the predicted disease index to the disease name
        idx_to_disease = {v: k for k, v in disease_to_idx.items()}
        predicted_disease_name = idx_to_disease[predicted_disease.item()]
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.write(f"Predicted Disease: {predicted_disease_name}")

    else:
        st.error("Please upload an image and select a species.")

# Tab 2 - Model Metrics

# Tab 3 - Visualize
