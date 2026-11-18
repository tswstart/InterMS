import json
import os
import subprocess

import numpy as np
import torch

import argparse
import logging
from datetime import datetime
from src.config import ConfigFile
import sys

# Save the dictionary dict to a JSON file
def dump_to_json(dict, path):
    if os.path.isfile(path):
        existing_json = True
    else:
        existing_json = False
    if existing_json:
        j = json.loads(open(path, encoding="utf-8").read())
    else:
        j = {}
    for k, v in dict.items():
        j[str(k)] = v
    with open(path, "w+", encoding="utf-8") as jFile:
        json.dump(j, jFile)

# Extract a set of columns (specified by keys) from DataFrame df and convert them to a NumPy array
def dataframe_raw_vectorization_with_numpy(df, keys: list):
    return np.stack([df[k].to_numpy() for k in keys], axis=-1).tolist()


def parse_params(config_file=None):
    try:
        if not config_file:
            parser = argparse.ArgumentParser(
                description="Parsing of config file name", formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

            parser.add_argument("-c", "--config", type=str, help="path to config file", default="configs/config.ini")
            args, unknown = parser.parse_known_args()
            if unknown != []:
                logging.warning(f"Unknown arguments {unknown}")
            config = vars(args)
            return ConfigFile(config["config"])
        else:
            return ConfigFile(config_file)
    except Exception:
        raise ImportError("Usage: pdbbind_pipeline.py [-c path_to_config.ini]")


def load_config(params, run_name=None):
    # basic logging config
    # params.base_name = os.path.splitext(os.path.split(config['config'])[1])[-2]
    if params.RUNINFO.run_name:
        params.base_name = params.RUNINFO.run_name
    else:
        params.base_name = "default"

    os.makedirs("outputs", exist_ok=True)

    # Set logging level to DEBUG
    logging.basicConfig(filename=os.path.join("outputs", "data_" + params.base_name + ".log"), level=logging.DEBUG)

    if run_name == sys.stdout:
        output_f = sys.stdout
    elif run_name:  # Overwrite run name indicated in config file
        output_f = open(os.path.join("outputs", "{}.log".format(run_name)), mode="a+")
    else:  
        raise  ValueError("Please provide a run name")

    print(f"\n\n============ RUN - {datetime.now()} ===================", file=output_f)

    return output_f

import gc

def add_small_graphs(protein_gm, compound_gm):
    """
    Return the transform for LinkNeighborLoader.
    Done inside the DataLoader worker process/thread:
      1. fetch the local subgraphs with n_id;
      2. assemble the Batch directly;
      3. move it to GPU non-blockingly;
    """
    def _transform(data):
       
        # -- Assemble the small graphs --
        data.pro_batch = protein_gm.get_batch(data['target'].n_id)
        data.lig_batch = compound_gm.get_batch(data['compound'].n_id)

        # Clean up the Python-level cache (recommended to call only inside the transform)
        gc.collect()

        return data
    
    return _transform


# Apply a weighted penalization to "false negative" cases (i.e. high actual value but low prediction) according to the given threshold
def penalize_false_negative(raw_loss, pred, target, threshold=4, penalization=3):
    pred_below = pred < threshold
    target_above = target > threshold
    return torch.where(pred_below & target_above, torch.mul(raw_loss, penalization), raw_loss)