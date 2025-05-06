import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors 
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
import re
import os
# from deepfashion_vanilla_att2 import gallery_dataset, image_root 


global_transform = transforms.Compose([ #transformation of images in a manner that is acceptable by ResNet18 (similar ImageNet format)
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

class TransformedGalleryDataset(Dataset): 
    def __init__(self, untransformed_gallery_dataset, transform=None):
        self.untransformed_gallery_dataset = untransformed_gallery_dataset
        self.transform = global_transform

    def __len__(self):
        return len(self.untransformed_gallery_dataset)

    def __getitem__(self, idx):
        raw_image, item_id = self.untransformed_gallery_dataset[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image, item_id

class DeepFashionModel(nn.Module):
    def __init__(self):
        super(DeepFashionModel, self).__init__()
        self.backbone = models.resnet18(weights=None) #pass images to resnet18 for feature extraction 
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])#cuts the classification and keeps the embeddings 
        self.fc = nn.Sequential( 
            nn.Linear(512, 256),#projects the outputed features from resnet18 from 512 to 256 dimensions
            nn.BatchNorm1d(256),
            nn.ReLU(), # introduce non-linearity, as images are non-linear
            nn.Linear(256, 256)) # produces final embedding of the image (numerical representation of image)

    def forward(self, x):
        x = self.backbone(x) #call resnet on image and produces feature tensor
        x = x.view(x.size(0), -1) #flattens tensor 
        x = self.fc(x) #call fully connected layer on flattened tensor to create a 256 embedding for the images
        return nn.functional.normalize(x, p=2, dim=1) #<<<Normalises length of each vector, this is in relation to tripletising the data

def load_model(model_path='trained_model_save.pth'): #loads the saved model weights
    model = DeepFashionModel()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    return model

def get_embedding(model, image_path): # transforms the image into something that ResNet18 can work with, this is also used as a sort of safety check to make sure the inputted image is of the desired format 
    transform = global_transform
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)
    return embedding.numpy()

def find_similar_items(query_embed, gallery_loader, model, gallery_entries, top_k=20): 
    gallery_embeddings = []
    gallery_paths = []
    gallery_entries = gallery_entries

    true_i = 0
    with torch.no_grad(): #and make it such that it returns more similar stuff, and also make it deal with id_similarity clashes, such that it only returns the id's, add an array and do that. hold up arr.append(list) if list already in arr then pass
        for images, _ in gallery_loader: #Returns images, skips over labels, since theres a discrepancy there with lenght of images vs length of labels array size wise
            embeddings = model(images)
            gallery_embeddings.append(embeddings.numpy())

            batch_size = len(images)
            gallery_paths.extend([gallery_entries[true_i + i][0] for i in range(batch_size)])
            true_i += batch_size     

        # print("Labels")  
        # print(gallery_ids)
        # print("Paths")
        # print(gallery_paths)
        
    gallery_embeddings = np.vstack(gallery_embeddings) #stack embeddings into 1 stack, this can't be done in the above as we also need to deal with gallery_entries, keeps embeddings, doesn't influence them.
    knn = NearestNeighbors(n_neighbors=top_k, metric='euclidean') #Defines NearestNeighbour class
    knn.fit(gallery_embeddings) #passes embeddings to set up space, fit sets up the data in a manner that can be used to search 
    distances, indices = knn.kneighbors(query_embed) #uses kneighbours to find neighbours of query_embed
    print(indices)
    return [(gallery_paths[i], distances[0][j]) for j, i in enumerate(indices[0])] #returns gallery paths and distances 

def display_ids_and_distances(similar_items):
    print("Similar Items with distance: ")
    blank_path = '/Users/abdullahalsalem/Desktop/'
    images_returned = []
    images_id = []
    for i, (img_path, dist) in enumerate(similar_items, 1):
        print(f"{i}. Item ID: {img_path} (Distance: {dist:.4f})")
        images_returned.append(blank_path + img_path)
        images_id.append(blank_path + img_path)

    print(images_id)
    paths_w_ids = []
    for i in images_id:
        match = re.match(r"(.+?/id_\d+/)", i)
        if match:
            paths_w_ids.append(match.group(1))
    print(f"All matched ID folders (with possible duplicates): {paths_w_ids}")
    print(f"Total IDs (with repeats): {len(paths_w_ids)}")
    no_repeating_ids = []
    for id in paths_w_ids:
        if id in no_repeating_ids:
            print("womp womp")  # duplicate detected
        else:
            no_repeating_ids.append(id)
    print("no repeating ids")
    print(no_repeating_ids)
    print(f"Unique IDs (no repeats): {no_repeating_ids}")
    print(f"Total unique: {len(no_repeating_ids)}")
    first_files = []
    for folder in no_repeating_ids:
        files = sorted(os.listdir(folder))
        if files:
            first_file = os.path.join(folder, files[0])
            first_files.append(first_file)
        else: 
            print(f"No files in folder: {folder}")
    print("first files")
    print(first_files)
    # print(images_returned) #/Users/abdullahalsalem/Desktop/img_highres/MEN/Denim/id_00006034/01_1_front.jp
    # return(images_returned)
    return(first_files)
