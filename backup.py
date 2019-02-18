import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
#http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf

from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim

from util import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=16, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=False, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_loader.dataset[i][0].permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.xlabel(class_names[test_loader.dataset[i][1]])


# define the model (a simple classifier)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        "block1"
        self.conv1 = nn.Conv2d(3, 16, 3)#inc channel 3->10, decrease the h,w size -3
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.subResConv=nn.Conv2d(3, 32, 5)
        self.subResbn=nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)#doesnt touch channel, decrease the h,w size -2

        "block2"
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.subResConv2=nn.Conv2d(32, 128, 5)
        self.subResbn2=nn.BatchNorm2d(128)

        "last"
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(in_features=256*3*3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=100)

    def forward(self, x):
        "block1" #F.relu()
        residual = x
        out = F.elu(self.bn1(self.conv1(x)+self.conv1(x)))
        out = F.elu(self.conv2(out))
        x = self.pool(self.subResbn(self.subResConv(residual)+out)) #residual

        "block2"
        residual = x
        out = F.elu(self.bn3(self.conv3(x)+self.conv3(x)))
        out = F.elu(self.conv4(out)+self.conv4(out))
        out=F.dropout(out, p=0.5, training=True)
        x = self.pool(self.subResbn2(self.subResConv2(residual)+out)) #residual


        x = F.relu(self.bn5(self.conv5(x)))#EXPLODING GRADIANTSSS
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



N = MyNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

# initialise the optimiser
# optimiser = torch.optim.SGD(N.parameters(), lr=0.001)

lr=0.01#0.05
optimiser = torch.optim.ASGD(N.parameters(), lr=lr, weight_decay=0.001)
epoch = 0
# liveplot = PlotLosses()


listOfAccu=[]
lrCounterOfDec=0
lrCounterOfInc=0

# criterion=nn.CrossEntropyLoss()

while (epoch<30):#10
    N = N.train()
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)
    train_acc_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(5000):#1000
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        optimiser.zero_grad()
        p = N(x)
        pred = p.argmax(dim=1, keepdim=True)
        loss = torch.nn.functional.cross_entropy(p, t)
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)
        train_acc_arr = np.append(train_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())

    N = N.eval()
    # iterate entire test dataset
    for x,t in test_loader:
        x,t = x.to(device), t.to(device)

        p = N(x)
        loss = torch.nn.functional.cross_entropy(p, t)
        pred = p.argmax(dim=1, keepdim=True)

        test_loss_arr = np.append(test_loss_arr, loss.cpu().data)
        test_acc_arr = np.append(test_acc_arr, pred.data.eq(t.view_as(pred)).float().mean().item())


    testacccc=round(test_acc_arr.mean()*100,2)
    resultsStr="""
    epoch_"""+str(epoch)+""":
            'train accuracy': """+str(round(train_acc_arr.mean()*100,2))+"""%,
            'val accuracy': """+str(testacccc)+"""%,
            'loss': """+str(round(train_loss_arr.mean(),2))+""",
            'val_loss': """+str(round(test_loss_arr.mean(),2))+"""
    --------------------------------------------------------------
    """
    print(resultsStr)


    if epoch>7:
        # print("isworking?",testacccc-listOfAccu[-1]<0.5 and lrCounterOfDec==1,testacccc-listOfAccu[-1]<0.5,lrCounterOfDec==1)
        if testacccc-listOfAccu[-1]<0.5 and lrCounterOfDec==1:
            lrCounterOfDec=0
            lrCounterOfInc=0
            lr-=lr*0.5
            print("DECREASED lrDec",lr)
            optimiser = torch.optim.ASGD(N.parameters(), lr=lr, weight_decay=0.001)
        elif testacccc-listOfAccu[-1] and lrCounterOfInc==3:
            lrCounterOfDec=0
            lrCounterOfInc=0
            lr+=lr*0.3
            print("INCREASED lrInc",lr)
            optimiser = torch.optim.ASGD(N.parameters(), lr=lr, weight_decay=0.001)
        elif testacccc-listOfAccu[-1]<0.5:
            lrCounterOfDec+=1
            lrCounterOfInc=0
            print("D:",lrCounterOfDec,"I:",lrCounterOfInc)
        elif testacccc-listOfAccu[-1]>=0.5:
            lrCounterOfDec=0
            lrCounterOfInc+=1
            print("D:",lrCounterOfDec,"I:",lrCounterOfInc)



    listOfAccu.append(testacccc)

    if testacccc>50:#46
        torch.save(N, "./v"+str(round(test_acc_arr.mean()*100,2))+"_t"+str(round(train_acc_arr.mean()*100,2))+"_e"+str(epoch)+".m")
    # elif testacccc<37 and epoch==10:
    #     raise SystemExit(0)
    # elif testacccc<42 and epoch==14:
    #     raise SystemExit(0)

    # if round(train_acc_arr.mean()*100,2)-round(test_acc_arr.mean()*100,2)>3:


    epoch = epoch+1

    #49.62
test_images, test_labels = next(test_iterator)
test_images, test_labels = test_images.to(device), test_labels.to(device)
test_preds = F.softmax(N(test_images).view(test_images.size(0), len(class_names)), dim=1).data.squeeze().cpu().numpy()

#torch.nn.functional
num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, test_preds, test_labels.cpu(), test_images.cpu().squeeze().permute(1,3,2,0).contiguous().permute(3,2,1,0))
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, test_preds, test_labels)

plt.show()
