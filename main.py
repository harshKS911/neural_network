import torch
import torch.nn as nn
from model import simplenn
from train import train_model
from utils.parser import get_args
from utils.logger import setup_logger
import os
import yaml 

if __name__== "__main__":
    logger=setup_logger()
    args=get_args()

    with open(args.config,'r') as f:
        config= yaml.safe_load(f)    
    logger.info("loaded hyperparameters sucessfully")


    model_cfg=config['model']
    model=simplenn(input=model_cfg["input"],hidden=model_cfg["hidden"],output=model_cfg["output"])
    
    logger.info("laoded the dimensions succesfully")

    trained_model=train_model(model,config,logger)

    #print(model.state_dict())


    
