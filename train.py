from model import simplenn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader , TensorDataset
import os
import numpy as np
import pandas as pd
from data import data


def save_model(model,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save(model.state_dict(),path)


def train_model(model, config,logger):
    lr=config["train"]["lr"]
    epoch=config["train"]["epochs"]
    batch=config["train"]["batch_size"]
   
    criterion=nn.MSELoss()
    optimizer=optim.SGD(model.parameters(),lr=lr)
    d=data()
    xt=d[0][1:800][:]
    
    yt=d[1][1:800][:]
    
    dataset= TensorDataset(xt,yt)
    loader= DataLoader(dataset,batch_size=batch,shuffle=True)
    
    for i in range(0,epoch):
        for xb,yb in loader:
            pred= model(xb)
            loss=criterion(pred,yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"epoch={i+1}/{epoch}, loss={loss.item():.4f}")
    
    logger.info("training done succesfully")
    save_model(model,config["path"])
    logger.info("model saved succesfully")
    return model


def test_model(model,config,logger):
    logger.info("testing results will be shown now")
    path="C:/Users/asus/OneDrive/Desktop/pirate_Ship/zself/save.pth"
    model.load_state_dict(torch.load(path,weights_only=True))
    model.eval()
    d=data()
    xte=d[0][800:][:] 
    yte=d[1][800:][:]
    dataset= TensorDataset(xte,yte)
    loader= DataLoader(dataset,batch_size=10)
    m1=[]
    m2=[]

    with torch.no_grad():
        for xb,yb in loader:
            pred=model(xb)
            m1.extend(pred.squeeze().tolist())
            m2.extend(yb.squeeze().tolist())


    return m1,m2 

    


    




        