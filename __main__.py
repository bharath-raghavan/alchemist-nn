import os
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import logging
_logger = logging.getLogger(__name__)
import asyncio

import numpy as np
import typer
app = typer.Typer()

import yaml
import json

from alchemist.runWrapper import RunWrapper
from alchemist.config import ConfigFile

Model = Annotated[Path,
                typer.Argument(help="NN Parameters for generation.")]

@app.command()
def train(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Train (or continue training of) a neural network
        flow model for generating structures.
    """
    hndl = RunWrapper(world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'), local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
    hndl.setup(ConfigFile.fromFile(config))
    hndl.train()

@app.command()
def generate(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Use a trained model to generate structures.
    """
    hndl = RunWrapper()
    self.setup(ConfigFile.fromFile(config), train=False)
    hndl.generate()

@app.command()
def createdb(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Process data and create a database. Can be used to run dynamics to generate structures.
    """
    with open(config, "r", encoding="utf-8") as f:
        if config.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    cofig_hdnl = ConfigFile(dataset=data)
    ds = cofig_hdnl.dataset.get()
    
app()