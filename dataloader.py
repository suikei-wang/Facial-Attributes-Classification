import os
import glob
import numpy as np
from matplotlib import pyplot
from PIL import Image
from zipfile import ZipFile 
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms

# define custom dataloader from torch
class celeba(Dataset):
  def __init__(self, data_path=None, label_path=None):
    self.data_path = data_path
    self.label_path = label_path

    # Data transforms
    self.transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   
  def __len__(self):
    return len(self.data_path)
  
  def __getitem__(self, idx):
    image_set = Image.open(self.data_path[idx])
    image_tensor = self.transform(image_set)
    image_label = torch.Tensor(self.label_path[idx])

    return image_tensor, image_label


def main():
	# specifying the zip file name 
	file_name = "/content/drive/My Drive/8536_project/CelebA/Img/img_align_celeba.zip"
	  
	# opening the zip file in READ mode 
	with ZipFile(file_name, 'r') as zip:  
	    if os.path.isdir('img_align_celeba') == 0:
	    	# extracting all the files 
	      	print('Extracting all the files now...') 
	      	zip.extractall() 
	      	print('Done!') 
	    else:
	      	print('File has already extracted.')

	data_path = sorted(glob.glob('img_align_celeba/*.jpg'))
	# print(len(data_path))


	# get the label of images
	label_path = "/content/drive/My Drive/8536_project/CelebA/Anno/list_attr_celeba.txt"
	label_list = open(label_path).readlines()[2:]
	data_label = []
	for i in range(len(label_list)):
		data_label.append(label_list[i].split())

	# transform label into 0 and 1
	for m in range(len(data_label)):
		data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
	  	data_label[m] = [int(p) for p in data_label[m]]

	# get the attributes names for display
	attributes = open(label_path).readlines()[1].split()

	dataset = celeba(data_path, data_label)
	# split data into train, valid, test set 7:2:1
	indices = list(range(202599))
	split_train = 141819
	split_valid = 182339
	train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	test_sampler = SubsetRandomSampler(test_idx)

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)

	validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler)

	testloader =  torch.utils.data.DataLoader(dataset, sampler=test_sampler)

	print(len(trainloader))
	print(len(validloader))
	print(len(testloader))

if __name__ == "__main__":
    main()