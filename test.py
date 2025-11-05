import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import data
from model import simplenn
from train import test_model
from utils.logger import setup_logger
from utils.parser import get_args
import yaml
import os

if __name__== "__main__":
    logger=setup_logger()
    args=get_args()

    with open(args.config,'r') as f:
        config= yaml.safe_load(f)    
    logger.info("loaded hyperparameters sucessfully")
    mcf=config["model"]

    model=simplenn(input=mcf["input"],hidden=mcf["hidden"],output=mcf["output"])

    k=test_model(model,config,logger)
    plt.scatter(k[0],k[1])
    plt.show()
    """
    predictions=np.array(k[0])
    targets=np.array(k[1])
    plt.hist([predictions,targets],bins=50,edgecolor="black")
    plt.show()
    """



