import argparse
#we define a mthod arg_parse which has argparse modeul from where we use argumentParser class to instantiate the object parser
#we add arguments for parsing which are passed as string at runtime and are compulsorily required
def get_args():
    parser= argparse.ArgumentParser(description="training of neural netwwork")
    parser.add_argument('--config',type=str,help='config.yaml for hyperparamter',required=True)
    parser.add_argument('--save_model',action='store_true',help='saved mdoel parmaters') 
    return parser.parse_args()