import os
import json

# Save species and disease mappings to JSON files
labels = [
    (i.split("___")[0], i.split("___")[1])
    for i in os.listdir("data/PlantVillage/segmented")
    if os.path.isdir("data/PlantVillage/segmented/" + i)
]
root_dir = "computer_vision/plant_diseases/data/PlantVillage/segmented"
species = sorted(list(set([i[0] for i in labels])))
diseases = sorted(list(set([i[1] for i in labels])))
species_to_idx = {species: idx for idx, species in enumerate(species)}
disease_to_idx = {disease: idx for idx, disease in enumerate(diseases)}

# write label mappings to JSON files
with open("species_to_idx.json", "w+") as f:
    json.dump(species_to_idx, f)
with open("disease_to_idx.json", "w+") as f:
    json.dump(disease_to_idx, f)
