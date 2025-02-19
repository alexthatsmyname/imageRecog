import os
import json

# Calea către folderul cu imagini
flowers_dir = os.path.join("data", "flowers")

# Calea către fișierul JSON cu mapping
json_path = "class_names.json"

# Încarcă mapping-ul din fișierul JSON
with open(json_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# Parcurge fiecare subfolder din flowers_dir
for subfolder in os.listdir(flowers_dir):
    subfolder_path = os.path.join(flowers_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Verifică dacă numele subfolderului există în mapping
        if subfolder in mapping:
            new_name = mapping[subfolder]
            new_path = os.path.join(flowers_dir, new_name)
            # Redenumește folderul
            os.rename(subfolder_path, new_path)
            print(f"Folderul '{subfolder}' redenumit în '{new_name}'")
        else:
            print(f"Folderul '{subfolder}' nu are mapping în JSON.")
