import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
def data():
    df=pd.read_csv("C:/Users/asus/OneDrive/Desktop/pirate_Ship/delaney-processed.csv")

    """
    plt.hist([(df["ESOL predicted log solubility in mols per litre"]),(df["measured log solubility in mols per litre"])],bins=50,edgecolor="black")
    plt.legend(["measured","predicted"])
    plt.show()
    print(f"shape of data \n {df.shape}")
    print(f"info about data \n {df.info} ")
    print(f"head of data frame\n {df.head()}") 

    """
    d=df.drop(columns=["measured log solubility in mols per litre","smiles","Compound ID","ESOL predicted log solubility in mols per litre"])
    x=torch.tensor(np.array(d),dtype=torch.float32)
    y=torch.tensor(np.array(df["ESOL predicted log solubility in mols per litre"]),dtype=torch.float32)
    
    return x,y
"""
d=data()
xt=d[0][1:700][:]
print(xt.shape)
"""
