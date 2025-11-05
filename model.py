import torch
import torch.nn as nn

class simplenn(nn.Module):
    def __init__(self,input,hidden,output):
        super(simplenn,self).__init__()
        #self.layer1= nn.Linear(input,hidden)
        #self.layer2=nn.Linear(hidden,final)
        #self.relu=nn.ReLU()
        layers=[]
        prev= input
        for i in hidden:
            layers.append(nn.Linear(prev,i))
            layers.append(nn.ReLU())
            prev=i

        layers.append(nn.Linear(prev,output))
        self.model=nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)
#i created class simplenn which inherits its featuers and attributes like neural architecture
#having linear projects and actiavtion fucnitons from nn.Module 
#then initiated the self instance of class with difffreent parameters as input
#using for loop we run through list of layers and do linear projection+activation through each layers
#made a list and passed by reference to the mthod forward whcih is also inherited from nn.module
#lets assume our layers are passed as list of hidden layers example [128,64]
