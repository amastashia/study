import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from  PIL import Image
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#Numpy to Tensor
def main01():
    np_array = np.array([[1,2,3],[4,5,6]])
    tc_array = torch.from_numpy(np_array)

#Network

#Sequential
def main02():
    model = nn.Sequential(
        nn.Conv2d(1,20,5),
        nn.ReLU(),
        nn.Conv2d(20,64,5),
        nn.ReLU()
    )

#nn.Module
def main03():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 64, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    
    model = Model()

#gradient
def main04():
    x = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    w = torch.tensor(2, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3, requires_grad=True, dtype=torch.float32)
    y=w*x+b
    y.backward()
    print(x.grad, w.grad, b.grad)

#use GPU
def main05():
    #Tensor
    x = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    x_gpu = x.to('cuda')

    #Model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 64, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
        
    model = Model()
    model.to('cuda')

#Loss function
def main06():
    #MSE Loss(for Regression)
    x = torch.randn(4)
    y = torch.randn(4)
    criterion = nn.MSELoss()
    loss = criterion(x,y)

    #CrossEntropy Loss(for Classification)
    x = torch.randn(1,4)
    y = torch.LongTensor([1]).random_(4)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(x,y)
    print(x)
    print(y)
    print(loss)

#Optimal function
def main07():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.lin1 = nn.Linear(in_features=10, out_features=10, bias=False)

        def forward(self, x):
            return self.lin1(x)
    
    def main():
        loss_list = []

        #pseudo dataset
        x = torch.randn(1, 10)
        w = torch.randn(1, 1)
        y = torch.mul(w, x) + 2

        #model
        model = Model()

        #loss function
        criterion = nn.MSELoss()

        #optimal function
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        #train
        for epoch in range(20):
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            loss_list.append(loss.data.item())
        
        plt.figure()
        plt.plot(loss_list)
        plt.grid()
        plt.show()


#Basic Neural Network for Image Classification
def main08():
    #dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 10
    height, width, channel = 32, 32, 3

    lr, momentum, weight_decay = 0.01, 0.9, 5e-4
    
    #model and functions
    class MLP(nn.Module):
        def __init__(self, hidden_size = 600):
            super(MLP, self).__init__()
            self.layer1 = nn.Linear(width*height*channel, hidden_size)
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, num_classes)
            
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = self.dropout1(x)
            x = F.relu(self.layer2(x))
            x = self.dropout2(x)
            x = F.relu(self.layer3(x))
            return x
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    #learning
    epochs = 50

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #train
        model.train()
        for images, labels in train_loader:
            images = images.view(-1,height*width*channel) #3d to 1d
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            train_loss += loss.item()
            train_acc += (labels_pred.max(1)[1]==labels).sum().item()
            loss.backward() #back propagate
            optimizer.step() #update weights
        avg_train_loss = train_loss/len(train_loader.dataset)
        avg_train_acc = train_acc/len(train_loader.dataset)

        #validation
        model.eval()
        with torch.no_grad(): #reduce RAM comsumption
            for images, labels in test_loader:
                images = images.view(-1,height*width*channel) #3d to 1d
                images = images.to(device)
                labels = labels.to(device)    
                labels_pred = model(images)
                loss  = criterion(labels_pred, labels)
                val_loss += loss.item()
                val_acc += (labels_pred.max(1)[1]==labels).sum().item()
        avg_val_loss = val_loss/len(test_loader.dataset)
        avg_val_acc = val_acc/len(test_loader.dataset)

        #log
        print(f'Epoch [{epoch+1}/{epochs}], train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    #plot
    plt.figure()
    plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.grid()

    plt.figure()
    plt.plot(range(epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.grid()

    plt.show()

#Convolutional Neural Network
def main09():
    #dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2)

    #parameters
    num_classes = 10
    lr, momentum, weight_decay = 0.01, 0.9, 5e-4
    epochs = 20

    #convolutional model
    class AlexNet(nn.Module):

        def __init__(self, num_classes=10) -> None:
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            train_loss += loss.item()
            train_acc += (labels_pred.max(1)[1]==labels).sum().item()
            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss/len(train_loader.dataset)
        avg_train_acc = train_acc/len(train_loader.dataset)

        #validation
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                val_loss += loss.item()
                val_acc += (labels_pred.max(1)[1]==labels).sum().item()
        avg_val_loss = val_loss/len(test_loader.dataset)
        avg_val_acc = val_acc/len(test_loader.dataset)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    

    #plot
    plt.figure()
    plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.grid()

    plt.figure()
    plt.plot(range(epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.grid()

    plt.show()

#CNN for another dataset you made
def main10():
    #parameters
    num_classes = 2
    lr, momentum, weight_decay = 0.01, 0.9, 5e-4
    batch_size = 5
    epochs = 20

    #define image preprocessing
    data_trainsforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    #preprocessing without normalization
    to_tensor_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    #data directory
    dir = 'data/hymenoptera_data'

    class CustomDataset(torch.utils.data.Dataset):
        classes = ['ant', 'bee']

        def __init__(self, dir, transform = None, train = True):
            #succeed to preprocessing class if you want
            self.transform = transform

            #list for image and label
            self.images = []
            self.labels = []

            #load train and valid data
            if train == True:
                ants_path = os.path.join(dir, 'train', 'ants')
                bees_path = os.path.join(dir, 'train', 'bees')
            else:
                ants_path = os.path.join(dir, 'val', 'ants')
                bees_path = os.path.join(dir, 'val', 'bees')

            #get images and label ant 0 and bee 1
            ant_images = glob.glob(ants_path+'/*.jpg')
            ant_labels = [0]*len(ant_images)

            bee_images = glob.glob(bees_path+'/*.jpg')
            bee_labels = [1]*len(bee_images)

            #unify 2 categories into one list
            for image, label in zip(ant_images,ant_labels):
                self.images.append(image)
                self.labels.append(label)
            
            for image, label in zip(bee_images,bee_labels):
                self.images.append(image)
                self.labels.append(label)

        def __getitem__(self, index):
            #get data by index
            image = self.images[index]
            label = self.labels[index]

            #load image from path (use PIL)
            with open(image, 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')
            
            #preprocessing
            if self.transform != None:
                image = self.transform(image)
            
            return image, label
        
        def __len__(self):
            #specify data size
            return len(self.images)
    
    #load dataset
    train_dataset = CustomDataset(dir, to_tensor_transforms, train=True)
    test_dataset = CustomDataset(dir, to_tensor_transforms, train=False)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    #anothor easier method
    """
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(dir, 'train'), data_trainsforms['train'])
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(dir, 'val'), data_trainsforms['val'])

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
    """

    #model
    class AlexNet(nn.Module):

        def __init__(self, num_classes) -> None:
            super(AlexNet, self).__init__()
            #convolutional process
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            #get features output dimention
            fc_input_example = torch.FloatTensor(batch_size, 3, 224, 224)
            fc_size = self.features(fc_input_example).view(fc_input_example.size(0), -1).size()[1]

            #classifing process
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        
    #same as model09
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for _ in tqdm(range(epochs)):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            train_loss += loss.item()
            train_acc += (labels_pred.max(1)[1]==labels).sum().item()
            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss/len(train_loader.dataset)
        avg_train_acc = train_acc/len(train_loader.dataset)

        #validation
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                val_loss += loss.item()
                val_acc += (labels_pred.max(1)[1]==labels).sum().item()
        avg_val_loss = val_loss/len(test_loader.dataset)
        avg_val_acc = val_acc/len(test_loader.dataset)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    

    #plot
    plt.figure()
    plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.grid()

    plt.figure()
    plt.plot(range(epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main10()