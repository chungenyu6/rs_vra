######################################################################################
# Standard library imports
import os
import argparse

# Third-party imports
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

# Local application imports
## Logger
from logger import logger
######################################################################################

# Declare global variables at the module level
args = None # arguments for system
argw = None # arguments for wandb

def init_arg():
    """Initialize the global arguments (args, argw)"""

    logger.info("Initializing the system arguments")
    global args, argw # use global keyword to modify the module-level variables

    # Arguments for system
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="random", required=True, type=str, help="Subset of POPE benchmark, only accepted values: random, popular, adversarial")
    parser.add_argument("--sample", default=1, required=False, type=int, help="Number of benchmark samples to test (1 - 300 samples)")
    parser.add_argument("--model", required=True, type=str, help="Model name, only accepted values: agent (ours), llava1.5")
    parser.add_argument("--max_reflexion_iters", default=1, required=False, type=int, help="Maximum number of reflexion iterations")
    parser.add_argument("--version", default="v1", required=False, type=str, help="Version of the model")
    parser.add_argument("--wandb", default=False, required=False, type=lambda x: str(x).lower() == 'true', help="Set to True to log to wandb")
    args = parser.parse_args()

    # Arguments for wandb
    argw = parser.parse_args()
    
    logger.info("System arguments initialized!")
    return args, argw

def init_login():
    """Initialize the system login"""

    logger.info("Initializing the system login")
    load_dotenv()

    # Login wandb
    if args.wandb == True:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            logger.error("WANDB_API_KEY not found in environment variables")
            raise ValueError("WANDB_API_KEY not found in environment variables")
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="POPE-n900",
            entity="chungenyu6-uwf",                        
            name=args.model,
            config={
                "version": "vra-rm1-vm1-aa1-ri1", # TODO: change to args.version later
                "subset": args.subset,
            }
        )
    
    # Login huggingface
    token = os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        logger.error("HUGGINGFACE_API_KEY not found in environment variables")
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
    login(token)

    logger.info("System login initialized!")
