import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

def create_deepfashion_datasets(gender, mapped_category,
    eval_partition_file='/Users/abdullahalsalem/Desktop/list_eval_partition_modified.txt',
    image_root='/Users/abdullahalsalem/Desktop'):
    
    print("Gender: ", gender)
    print("Mapped_category: ", mapped_category)

    def parse_eval_partition(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]  # skips the first 2 lines
        return [(path, item_id, split) for path, item_id, split in (line.strip().split() for line in lines)]

    class DeepFashionDataset(Dataset):
        def __init__(self, base_dir, entries):
            self.base_dir = base_dir
            self.entries = entries

        def __len__(self): # Gives the number of entries
            return len(self.entries)

        def __getitem__(self, idx): # Generates and gets the specific image entry 
            img_path, item_id, _ = self.entries[idx]  # assigns img_path = images path from the entry, assigns item_id from entry // id_x is the id of the entry
            full_path = os.path.join(self.base_dir, img_path)  # creates the images path to access by combining directory with img_path
            image = Image.open(full_path).convert('RGB') #opens image using Image API, and converts it into RGB format, 
            return image, item_id

    all_entries = parse_eval_partition(eval_partition_file) #extracts the entries using parse_eval, as well as the number of entries 
    
    train_entries = [e for e in all_entries if e[2] == 'train'] #splits entries to "train" set, so model can train on this data specifically (checks this by checking if the e[2], third index in array is train, query, or gallery)
    query_entries = [e for e in all_entries if e[2] == 'query']#splits entires to "query" set, so model can run validation tests on them specifically 
    
    gallery_entries = [e for e in all_entries if e[2] == 'gallery' and e[0].startswith(f'img_highres/{gender}/{mapped_category}')] #splits entries to "gallery" set, so modal can have data it's not seen to be tested on by developer (me) // Also filters the Gender and the Category, such that searches can be more time-efficient
    
    train_dataset = DeepFashionDataset(image_root, train_entries) #create training dataset
    query_dataset = DeepFashionDataset(image_root, query_entries) #create query dataset
    gallery_dataset = DeepFashionDataset(image_root, gallery_entries) #create gallery dataset

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Query size: {len(query_dataset)}")
    print(f"Gallery size: {len(gallery_dataset)}")
    print(f"First image ID: {train_dataset[0][1]}")
    print(f'img_highres/{gender}/{mapped_category}')

    return (gallery_dataset, gallery_entries) #returns gallery_dataset and gallery_entries from eval_partition to send to barebones_model.py

#----------The below code, is the commented version of the above, with the distinction that the above is wrapped in a function, so that it can be used in the flask_app.py file--------------------#

# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms
# # from categoriser import (
# #     gender, 
# #     mapped_category
# # )


# def parse_eval_partition(file_path): 
#     with open(file_path, 'r') as f:
#         lines = f.readlines()[2:]  
#     data = []
#     for line in lines:
#         path, item_id, split = line.strip().split()
#         data.append((path, item_id, split))
#     return data

# class DeepFashionDataset(Dataset): 
#     def __init__(self, base_dir, entries):# Constructor <<ORIGINAL CODE transform = None
#         self.base_dir = base_dir
#         self.entries = entries

#     def __len__(self): # Gives the number of entries
#         return len(self.entries)

#     def __getitem__(self, idx): # Generates and gets the specific image entry 
#         # img_path, item_id, _ = self.entries[idx] # assigns img_path = images path from the entry, assigns item_id from entry // id_x is the id of the entry #<<ORIGINAL CODE,UNCOMMENT IF NOT WORKING
#         img_path = self.entries[idx][0] #assigns images path by taking it from entry in eval_part_list
#         item_id = self.entries[idx][1] #assigns images id by taking it from entry in eval_part_list 
#         # _ = self.entries[idx][2]
#         full_path = os.path.join(self.base_dir, img_path) # creates the images path to access by combining directory with img_path
#         image = Image.open(full_path).convert('RGB') #opens image using Image API, and converts it into RGB 
#         return image, item_id 

# eval_partition_file = '/Users/abdullahalsalem/Desktop/list_eval_partition_modified.txt' #path for list_eval_partition.txt
# image_root = '/Users/abdullahalsalem/Desktop' #base root directory 

# all_entries = parse_eval_partition(eval_partition_file) #extracts the entries using parse_eval, as well as the number of entries 

# # gender = input("Input gender: ")
# # print(f"You entered: {gender}")

# # category = input("Input category: ")
# # print(f"Your category is: {category}")
# # gender = 
# # category = 

# train_entries   = [e for e in all_entries if e[2] == 'train'] #splits entries to "train" set, so model can train on this data specifically (checks this by checking if the e[2], third index in array is train, query, or gallery)
# query_entries   = [e for e in all_entries if e[2] == 'query'] #splits entires to "query" set, so model can run validation tests on them specifically 

# gallery_entries = [e for e in all_entries if e[2] == 'gallery'] #splits entries to "gallery" set, so modal can have data it's not seen to be tested on by developer (me)

# train_dataset   = DeepFashionDataset(image_root, train_entries) #create training dataset
# query_dataset   = DeepFashionDataset(image_root, query_entries) #create querying dataset
# gallery_dataset = DeepFashionDataset(image_root, gallery_entries) #create gallery dataset

# batch_size = 128 

# train_loader   = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #Want training data to be random so model would learn 
# query_loader   = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
# gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

# print(f"TESTING: Train size:   {len(train_dataset)}")
# print(f"TESTING: Query size:   {len(query_dataset)}")
# print(f"TESTING: Gallery size: {len(gallery_dataset)}")
# print(f"TESTING: Total Size: {len(train_dataset) + len(query_dataset) + len(gallery_dataset)}")

# sample_image_id = train_dataset[0][1] #getting sample to print
# print(f"TESTING: 1st Image itemid: {sample_image_id}") 
# print("deepfashion_vanilla called")