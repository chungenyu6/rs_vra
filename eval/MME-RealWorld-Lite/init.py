######################################################################################
# Standard library imports
import os
import argparse

# Third-party imports
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

# Local imports
from logger import logger, setup_logger

# Declare global variables at the module level
args = None # arguments for system
argw = None # arguments for wandb

def init_arg():
    """Initialize the global arguments (args, argw)"""
    
    print("Initializing the system arguments")
    
    # Arguments for system
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2category", default="attribute_motion_multipedestrians", required=True, type=str, help="L2-Category of MME-RealWorld-Lite benchmark, only accepted values: attribute_motion_multipedestrians, attribute_motion_multivehicles, attribute_motion_pedestrian, attribute_motion_vehicle, attribute_visual_trafficsignal, object_count, objects_identify, person_attribute_color, person_attribute_orientation, person_counting, vehicle_attribute_color, vehicle_attribute_orientation, vehicle_counting, vehicle_location, color, count, position, attention_trafficsignal, prediction_intention_ego, prediction_intention_pedestrian, prediction_intention_vehicle, relation_interaction_ego2pedestrian, relation_interaction_ego2trafficsignal, relation_interaction_ego2vehicle, relation_interaction_other2other, calculate, intention, property")
    parser.add_argument("--sample", default=1, required=False, type=int, help="Number of samples for evaluation; (1-100 samples) for diagram, table; (1-19 samples) for person/attribute/orientation; (1-50 samples) otherwise")
    parser.add_argument("--model", required=True, type=str, help="Model name, only accepted values: agent, llava15, geochat, gemma3, mistral31") # NOTE: change this if needed
    parser.add_argument("--version", default="v1", required=False, type=str, help="Version of the model") # NOTE: change this if needed
    parser.add_argument("--max_reflexion_iters", default=1, required=False, type=int, help="Maximum number of reflexion iterations")
    parser.add_argument("--wandb", default=False, required=False, type=lambda x: str(x).lower() == 'true', help="Set to True to log to wandb")
    args = parser.parse_args()

    # Arguments for wandb
    argw = parser.parse_args()
    
    print("System arguments initialized!")
    return args, argw

def init_logger(args):
    """Initialize the logger with the current args"""
    return setup_logger(args)

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
            project="MME-RealWorld-Lite",
            entity="chungenyu6-uwf",
            name=args.model,
            config={
                "version": args.version,
                "l2category": args.l2category,
            }
        )
    
    # Login huggingface
    token = os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        logger.error("HUGGINGFACE_API_KEY not found in environment variables")
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
    login(token)

    logger.info("System login initialized!")

def initialize():
    """Main initialization function that sets up everything in the correct order"""
    global args, argw
    args, argw = init_arg()
    init_logger(args)
    init_login()
    return args, argw, logger
