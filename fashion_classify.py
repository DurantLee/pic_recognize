import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

train_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.linear_relu_stack=nn.Sequential(
            nn.Conv2d(1,10,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10,20,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20,40,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(40*4*4,128),
            nn.Linear(128,10)
        )
    def forward(self,x):
        logits=self.linear_relu_stack(x)
        return logits
    
batch_size=64
learning_rate=5e-3
epochs=5
momentum=0.10


model=NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)

def train(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    model.train()
    
    for batch, (data,target) in enumerate(dataloader):
        pred=model(data)
        optimizer.zero_grad()
        loss=loss_fn(pred,target)

        loss.backward()
        optimizer.step()
        
        if batch%100==0:
            loss, current=loss.item(), batch*batch_size+len(data)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader,model):
    model.eval()
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct=0,0

    with torch.no_grad():
        for data, target in dataloader:
            pred=model(data)
            test_loss+=loss_fn(pred,target).item()
            correct+= (pred.argmax(1)==target).type(torch.float).sum().item()

    test_loss/=num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    #train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model)
print("Well_Done!")
torch.save(model.state_dict(), 'model.pth')