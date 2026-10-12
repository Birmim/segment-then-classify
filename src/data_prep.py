import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

def images_prep(folder_path, size=(128, 128)):
    raw_data = []
    data_names = [] 

    path = os.path.join(folder_path, "images", "images")
    for img_name in sorted(os.listdir(path)):
        if  img_name.startswith(".") or not (img_name.endswith(".jpg")) :
            continue

        full_path = os.path.join(path, img_name)
        image = Image.open(full_path).convert("RGB").resize(size)
        image_as_vector = np.array(image)
        raw_data.append(image_as_vector)
        data_names.append(img_name)

    normalised_data = np.array(raw_data).astype(np.float32) / 255.0  # Shape (N img in the dataset, 128, 128, 3) 

    raw_data = torch.from_numpy(normalised_data).permute(0, 3, 1, 2) # Shape (N img in the dataset, 3, 128, 128)

    return raw_data, data_names # Send back a normalized images tensor and images names 

def masks_prep(folder_path, data_names, size=(128, 128)):
    trimap = []

    path = os.path.join(folder_path, "annotations", "annotations", "trimaps")
    for item in data_names:
        item = os.path.splitext(item)[0] + ".png"
    
        trimap_path = os.path.join(path, item)
        trimap.append(np.array(Image.open(trimap_path).resize(size, resample=Image.NEAREST)))

    trimap = np.array(trimap)
    trimap[trimap == 3] = 1
    trimap[trimap == 2] = 0

    return torch.from_numpy(trimap).unsqueeze(1).float() 
    

def change_names_to_species(data_names):
    breeds_dict = {}

    for animal_name in data_names:
        if animal_name[0].isupper():
            breeds_dict[animal_name] = 0
        else:
            breeds_dict[animal_name] = 1
    
    return np.array(list(breeds_dict.values()))


def split_dataset_idx(images_breeds, val_size=0.1, test_size=0.1, random_state=42):
    images_idx = np.arange(len(images_breeds)) # grab the dataset size and transform it into a sequential list

    # This split the index into train, val and test then all data can match their index
    idx_train, idx_temp = train_test_split(
        images_idx,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=images_breeds
    )

    relative_test_size = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=relative_test_size,
        random_state = random_state,
        stratify=images_breeds[idx_temp]
    )

    return idx_train, idx_val, idx_test

def split_data(folder_path):
    images, images_names = images_prep(folder_path)
    masks = masks_prep(folder_path=folder_path, data_names=images_names)
    species = change_names_to_species(images_names)

    # Split the indexes
    idx_train, idx_val, idx_test = split_dataset_idx(species)

    X_train = images[idx_train]
    X_val = images[idx_val]
    X_test = images[idx_test]

    y_cnn_train = species[idx_train]
    y_cnn_val = species[idx_val]
    y_cnn_test = species[idx_test]

    y_unet_train = masks[idx_train]
    y_unet_val = masks[idx_val]
    y_unet_test = masks[idx_test]

    return X_train, X_val, X_test, y_cnn_train, y_cnn_val, y_cnn_test, y_unet_train, y_unet_val, y_unet_test
     
