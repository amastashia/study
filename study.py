import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

    if __name__ == "__main__":
        main()




if __name__ == "__main__":
    main7()