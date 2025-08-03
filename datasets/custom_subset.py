import torch
import numpy as np

import torchvision.transforms as T 
# class Subset(torch.utils.data.Dataset): # for fixed location
#     def __init__(self, dataset, indices, transform, device='cuda'):
#         self.dataset = dataset
#         self.indices = indices
#         self.transform = transform
#         self.targets = np.array(dataset.targets)[self.indices]
#         self.device = device
        
#         ratio = 0.2
#         self.size = int(np.sqrt(ratio*224*224))
#         print('size',ratio,self.size)
#         self.mask = np.random.randint(low=0, high=224-self.size, size=(len(self.indices),2))  #83/124
        

#         self.post_transform = []
#         self.pre_transform = []
#         idx = 0
#         for trans in transform.transforms:
#             if idx<2:
#                 self.pre_transform.append(trans)
#             else:
#                 self.post_transform.append(trans)
#             idx +=1
#         self.post_transform = T.Compose(self.post_transform)
#         self.pre_transform = T.Compose(self.pre_transform)

#     def __getitem__(self, idx):
#         im, targets = self.dataset[self.indices[idx]]
#         mask = self.mask[self.indices[idx]]

#         # print('testttttttt')
        
#         if self.transform:
#             im = self.pre_transform(im)
#             im[0,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] =  0.485
#             im[1,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] = 0.456 
#             im[2,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] =  0.406
#             im = self.post_transform(im)
#             # print('testttttttt')
#             # exit()
#         return im, targets

#     def __len__(self):
#         return len(self.indices)

# class Subset(torch.utils.data.Dataset): 
#     # for fixed location + boundary location
#     def __init__(self, dataset, indices, transform, device='cuda'):
#         self.dataset = dataset
#         self.indices = indices
#         self.transform = transform
#         self.targets = np.array(dataset.targets)[self.indices]
#         self.device = device
        
#         ratio = 0.2
#         self.size = int(np.sqrt(ratio*224*224))
#         print('size',ratio,self.size)
#         # location is not in r/4 -> 3r/4 where r is the size of image.         
#         self.mask = np.random.randint(low=0, high=224-self.size, size=(len(self.indices),2)) 
#         test =0 
#         for i in range(len(self.indices)):
#             # if 56 <= self.mask[0]  <= 168 and 56 <= self.mask[1]  <= 168:
#             while 56 <= self.mask[i][0]  <= 224-56-self.size and 56 <= self.mask[i][1]  <= 224-56-self.size:
#                 self.mask[i][0] = np.random.randint(0, 224-self.size)
#                 self.mask[i][1] = np.random.randint(0, 224-self.size)
#                 test+=1

#             # print(i, self.mask[i])

#         print(self.mask[:20],test)
#         # exit()
#         self.post_transform = []
#         self.pre_transform = []
#         idx = 0
#         for trans in transform.transforms:
#             if idx<2:
#                 self.pre_transform.append(trans)
#             else:
#                 self.post_transform.append(trans)
#             idx +=1
#         self.post_transform = T.Compose(self.post_transform)
#         self.pre_transform = T.Compose(self.pre_transform)

#     def __getitem__(self, idx):
#         im, targets = self.dataset[self.indices[idx]]
#         mask = self.mask[self.indices[idx]]

#         # print('testttttttt')
        
#         if self.transform:
#             im = self.pre_transform(im)
#             im[0,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] =  0.485
#             im[1,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] = 0.456 
#             im[2,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] =  0.406
#             im = self.post_transform(im)
#             # print('testttttttt')
#             # exit()
#         return im, targets

#     def __len__(self):
#         return len(self.indices)



# class Subset(torch.utils.data.Dataset): 
#     # for random location + boundary location
#     def __init__(self, dataset, indices, transform, device='cuda'):
#         self.dataset = dataset
#         self.indices = indices
#         self.transform = transform
#         self.targets = np.array(dataset.targets)[self.indices]
#         self.device = device
        
#         ratio = 0.2
#         self.size = int(np.sqrt(ratio*224*224))
#         print('size',ratio,self.size)
#         # location is not in r/4 -> 3r/4 where r is the size of image.         
       

#     def __getitem__(self, idx):
#         im, targets = self.dataset[self.indices[idx]]
        
#         mask = np.random.randint(low=0, high=224-self.size, size=2) 
#         while 56 <= mask[0]  <= 224-56-self.size and 56 <= mask[1]  <= 224-56-self.size:
#             mask[0] = np.random.randint(0, 224-self.size)
#             mask[1] = np.random.randint(0, 224-self.size)
            
#         # print('testttttttt')
        
#         if self.transform:
#             im = self.transform(im)
#             im[0,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] =  0.485
#             im[1,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] = 0.456 
#             im[2,mask[0]:mask[0]+self.size, mask[1]:mask[1]+self.size] =  0.406
            
#         return im, targets

#     def __len__(self):
#         return len(self.indices)





class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = np.array(dataset.targets)[self.indices]

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, targets

    def __len__(self):
        return len(self.indices)

class SingleClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_class):
        self.dataset = dataset
        self.indices = np.where(np.array(dataset.targets) == target_class)[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.target_class = target_class

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)


class ClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_classes):
        self.dataset = dataset
        self.indices = np.where(
            np.isin(np.array(dataset.targets), np.array(target_classes)))[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.target_classes = target_classes

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)
