import torch
import torchvision
import numpy as np 
from PIL import Image 
import os
from torchvision import transforms
import pandas as pd


from torch.utils.data import Dataset



tfms_normal = transforms.Compose([
    transforms.CenterCrop(size=(256,256)), 
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.46,0.44,0.39], std= [0.19,0.18,0.19])
])

tfms_target = transforms.CenterCrop(size = (256,256))
    

class Data_provider_SYSU(Dataset):


    def __init__(self, path):
        self.data_path = path 
        self.pre_path = os.path.join(path, "time1")
        self.post_path = os.path.join(path, "time2")
        self.target_path = os.path.join(path, "label")

    def __len__(self):
        return len(os.listdir(self.post_path))
    
    def __getitem__(self, idx):

        post_list = os.listdir(self.pre_path)
        pre_list = os.listdir(self.post_path)
        target_list = os.listdir(self.target_path)

        post_list.sort()
        pre_list.sort()
        target_list.sort()

        pre_image_path = os.path.join(self.pre_path, pre_list[idx])
        post_image_path = os.path.join(self.post_path, post_list[idx])
        target_path =  os.path.join(self.target_path, target_list[idx])

        pre_image = Image.open(pre_image_path)
        post_image = Image.open(post_image_path)
        target_image = Image.open(target_path) 

        pre_image = tfms_normal(pre_image)
        post_image = tfms_normal(post_image)


        target_image = torch.tensor(np.array(tfms_target(target_image))/255).long()



        return pre_image, post_image, target_image
    

class Data_provider_levir(Dataset):


    def __init__(self, path):
        self.data_path = path 
        self.pre_path = os.path.join(path, "A")
        self.post_path = os.path.join(path, "B")
        self.target_path = os.path.join(path, "label")

    def __len__(self):
        return len(os.listdir(self.post_path))
    
    def __getitem__(self, idx):

        pre_list = os.listdir(self.pre_path)
        post_list = os.listdir(self.post_path)
        target_list = os.listdir(self.target_path)

        post_list.sort()
        pre_list.sort()
        target_list.sort()

        pre_image_path = os.path.join(self.pre_path, pre_list[idx])
        post_image_path = os.path.join(self.post_path, post_list[idx])
        target_path =  os.path.join(self.target_path, target_list[idx])
        # print(pre_image_path)
        # print(post_image_path)
        # print(target_path)

        pre_image = Image.open(pre_image_path)
        post_image = Image.open(post_image_path)
        target_image = Image.open(target_path) 

        pre_image = tfms_normal(pre_image)
        post_image = tfms_normal(post_image)


        target_image = torch.tensor(np.array(target_image)/ 255).long()



        return pre_image, post_image, target_image
    
    




class Data_provider_WHU(Dataset):


    def __init__(self, path, file):
        self.data_path = path 
        self.pre_path = os.path.join(path, "A")
        self.post_path = os.path.join(path, "B")
        self.target_path = os.path.join(path, "label")
        self.data_names = np.array(pd.read_csv(file, names=["tt"]))


    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):

        name = self.data_names[idx].item()

        # post_list = os.listdir(self.post_path)
        # pre_list = os.listdir(self.pre_path)
        # target_list = os.listdir(self.target_path)


        # post_list.sort()
        # pre_list.sort()
        # target_list.sort()



        pre_image_path = os.path.join(self.pre_path,name )
        post_image_path = os.path.join(self.post_path,name )
        target_path =  os.path.join(self.target_path, name)



 

        pre_image = Image.open(pre_image_path)
        post_image = Image.open(post_image_path)
        target_image = Image.open(target_path) 


        pre_image = tfms_normal(pre_image)
        post_image = tfms_normal(post_image)

        target_image = torch.tensor(np.array(tfms_target(target_image)) / 255).long()
  



        return pre_image, post_image, target_image











    




