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

from .config import load_dict, DynamicConfig
from .data.lj import LJDataset
from .dynamic import simulate_system
from .asedb import to_ase, add_mols

Model = Annotated[Path,
                typer.Argument(help="NN Parameters for generation.")]

@app.command()
def train(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Train (or continue training of) a neural network
        flow model for generating structures.
    """
    hndl = Main()
    self.setup_dataset()
    self.setup_model()
    self.setup_optim()
    hndl.train()

@app.command()
def generate(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Use a trained model to generate structures.
    """
    hndl = Main()
    hndl.config_file.dataset = hdnl.config_file.generate
    self.setup_dataset()
    self.setup_model()
    hndl.generate()

@app.command()
def dumpdb(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Run dynamics on a model to generate structures.
    """
    hndl = Main()
    hndl.setup_dataset()