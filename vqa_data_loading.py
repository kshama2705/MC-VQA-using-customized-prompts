import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from vqav1_dataloader import VQAv1Dataset
from vqav2_dataloader_yesno import VQAv2Dataset,collater


image_transform=T.Compose([T.ToTensor(),T.Resize((640,480))])
vqa_dataset=VQAv2Dataset(root="dataset",image_transforms=image_transform)



#train_dataloader = DataLoader(vqa_dataset, batch_size=1, shuffle=True)

train_dataloader = DataLoader(vqa_dataset, batch_size=1, shuffle=True)

print(" No of Multiple Choice Questions in VQA V2 dataset")
print(len(train_dataloader))

out=next(iter(train_dataloader))

for key,value in out.items():

    print(key) 
    print(value) 
    #print(value.shape)

