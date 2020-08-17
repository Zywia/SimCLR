import torch
import torch.nn as nn
import sys
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import importlib.util
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

folder_name = 'runs/Aug17_10-02-31_WMII2084'

checkpoints_folder = os.path.join(folder_name, 'checkpoints')
config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))


# Load the neural net module
spec = importlib.util.spec_from_file_location("model", 'models/resnet_simclr.py')
resnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_module)

model = resnet_module.ResNetSimCLR(**config['model'])
model.eval()

state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


class SimCLRTrasfer(nn.Module):

    def __init__(self, pretrained_model):
        super(SimCLRTrasfer, self).__init__()
        self.pretrained_model = pretrained_model

        
        self.features = nn.Sequential(*list(self.pretrained_model.children())[:-2])

        # projection MLP
        self.l1 = nn.Linear(2048, 1000)
        
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        
        return x




transfer_model = SimCLRTrasfer(model)


IMAGENET_PATH = "/media/data2/infotech/datasets/imagenet/"
TRAIN_FOLDER_NAME = "train"
VALIDATION_FOLDER_NAME = "val"

LIST_OF_FOLDERS_SLASH_CLASSES = os.listdir(os.path.join(IMAGENET_PATH, TRAIN_FOLDER_NAME))


def extract_images_and_their_classes(train_or_val):   
    list_of_images_paths = { x: os.listdir(os.path.join(IMAGENET_PATH, train_or_val, x))
    for x in LIST_OF_FOLDERS_SLASH_CLASSES}

    this_is_going_out = np.array( [(z, x)  for x, y 
    in list_of_images_paths.items() for z in y] )

    shake_it = np.random.permutation(this_is_going_out.shape[0])

    return this_is_going_out[shake_it]


def generator(b_s, phase_gen='train'):
    def add_path_information(image_and_class):
        return np.array(os.path.join(IMAGENET_PATH, phase_gen, image_and_class[1], image_and_class[0]), dtype=object)
    
    images_and_their_classes = extract_images_and_their_classes(phase_gen).astype(object)
   
    counter = 0
    while True:
        images = np.apply_along_axis(add_path_information, axis=1, 
        arr=images_and_their_classes[counter:counter + b_s])

        y = enc.transform(images_and_their_classes[counter:counter + b_s, 1].reshape(-1, 1)).toarray()
        
        yield preprocess_images(images, shape_r, shape_c), y
        counter = (counter + b_s) % images_and_their_classes.shape[0]


import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer


class ImagenetDataset(Dataset):
    def __init__(self, dataset_part:str, labels_part:float = 1., transform = None):
        self.__transforms = transform
        self.__imagenet_path = "/media/data2/infotech/datasets/imagenet/"
        self.__LIST_OF_FOLDERS_SLASH_CLASSES = np.array( os.listdir(
            os.path.join(IMAGENET_PATH, dataset_part)))
        self.__dataset_part = dataset_part
        list_of_images_paths = { x: np.array(os.listdir(os.path
                                               .join(self.__imagenet_path, dataset_part, x)))
                                [:int (len(os.listdir(os.path
                                               .join(self.__imagenet_path, dataset_part, x)))
                                       * labels_part)]
                                for x in self.__LIST_OF_FOLDERS_SLASH_CLASSES}

        this_is_going_out = np.array( [(z, x)  for x, y
                                       in list_of_images_paths.items() for z in y] )

        shake_it = np.random.permutation(this_is_going_out.shape[0])
        
        self.__images = this_is_going_out[shake_it]
        
        self.enc = preprocessing.OneHotEncoder()
        self.enc.fit(self.__LIST_OF_FOLDERS_SLASH_CLASSES.reshape(-1,1))
        
    def __len__(self):
        return self.__images.shape[0]
    
    def __getitem__(self, idx):
        to_work_on = self.__images[idx]
        
        path_to_image = os.path.join(self.__imagenet_path, self.__dataset_part, 
                                     to_work_on[1], to_work_on[0]) 
        image = Image.open(path_to_image).convert('RGB')
        one_hot = self.enc.transform(to_work_on[1].reshape(-1, 1)).toarray()
        
        one_hot = np.where(one_hot==1)[1][0]
        
        
        if self.__transforms is not None:
            image = self.__transforms(image)
        
        return image, torch.tensor(one_hot, dtype=torch.long)
        
        
batch_size = 128

transform = transforms.Compose(
    [transforms.Resize( (224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

optimizer = torch.optim.SGD(transfer_model.parameters(), lr=0.05 * batch_size / 256, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()

train_dataset = ImagenetDataset("train", .1, transform=transform)
val_dataset = ImagenetDataset("val", .1, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

transfer_model = transfer_model.to(device)

def check_accuracy(model, data_loader, criterion):
    with torch.no_grad(): 
        model.eval()
        
        top_five_answer = 0
        top_one_anwer = 0
        valid_loss = 0.0
        counter = 0
        
        for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model_resoult = model(inputs)
                loss = criterion(model_resoult, labels)
                
                valid_loss += loss.item()
                counter += 1
                
                model_resoult, indices = torch.sort(model_resoult, dim=1, descending=True)

                indices = indices[:, :5]

                resoult = indices == labels.reshape(-1, 1)
                resoult_top_one =  indices[:,0] == labels
                top_five_answer += torch.sum(resoult)
                top_one_anwer += torch.sum(resoult_top_one)
        valid_loss /= counter
    model.train()
    return (top_five_answer.float() / (batch_size * len(val_loader)) * 100, 
            top_one_anwer.float() / (batch_size * len(val_loader)) * 100,
           valid_loss)
            

def validate( model, valid_loader, criterion):

    # validation steps
    with torch.no_grad():
        model.eval()

        valid_loss = 0.0
        counter = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model_resoult = model(inputs)
            loss = criterion(model_resoult, labels)
            
            valid_loss += loss.item()
            counter += 1
        valid_loss /= counter
    model.train()
    return valid_loss


n_iter = 0
best_valid_loss = np.inf
tensorboard_writer = SummaryWriter()
os.makedirs(os.path.join(tensorboard_writer.log_dir, 'checkpoints'))

for epoch in range(30):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
#         print(i)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = transfer_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 200 == 199:    # print every 2000 mini-batches
            tensorboard_writer.add_scalar("Loss/train", running_loss / 200, global_step=n_iter)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0            
    
        n_iter += 1
        
    top_five, top_one, valid_loss = check_accuracy(transfer_model, val_loader, criterion)
    tensorboard_writer.add_scalar("Acc/val/top_five", top_five, epoch)
    tensorboard_writer.add_scalar("Acc/val/top_one", top_one, epoch)
    
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(transfer_model.state_dict(),  os.path.join(tensorboard_writer.log_dir, 'checkpoints', 'model.pth'))
        
print('Finished Training')