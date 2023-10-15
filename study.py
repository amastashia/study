import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#Numpy to Tensor
def main1():
    np_array = np.array([[1,2,3],[4,5,6]])
    tc_array = torch.from_numpy(np_array)

#Network

#Sequential
def main2():
    model = nn.Sequential(
        nn.Conv2d(1,20,5),
        nn.ReLU(),
        nn.Conv2d(20,64,5),
        nn.ReLU()
    )

#nn.Module
def main3():
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
def main4():
    x = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    w = torch.tensor(2, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3, requires_grad=True, dtype=torch.float32)
    y=w*x+b
    y.backward()
    print(x.grad, w.grad, b.grad)

#use GPU
def main5():
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
def main6():
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
def main7():
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
def main8():
    #dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 10
    height, width, channel = 32, 32, 3

    lr, momentum, weight_decay = 0.01, 0.9, 5e-4
    
    #model and functions
    class MLP(nn.module):
        def __init__(self, hidden_size = 600):
            super(MLP, self).__init__()
            self.layer1 = nn.Linear(width*height*channel, hidden_size)
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, num_classes)
            
            self.dropout1 = nn.Dropout2d(0.2)
            self.dropout2 = nn.Dropout2d(0.2)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = self.dropout1(x)
            x = F.relu(self.layer2(x))
            x = self.dropout2(x)
            x = F.relu(self.layer3(x))
            return x
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        for i, (images, labels) in enumerate(train_loader()):
            images = images.view(-1,height*width*channel) #3d to 1d
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            train_loss += loss.item()
            train_acc += (labels_pred.max(1)[1]==labels).sum().item()
            


if __name__ == "__main__":
    main7()