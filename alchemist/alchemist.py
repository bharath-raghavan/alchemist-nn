from pathlib import Path
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import logging
_logger = logging.getLogger(__name__)

import typer
app = typer.Typer()

from .config import load_dict, DynamicSetup
from .data.lj import LJDataset

Model = Annotated[Path,
                typer.Argument(help="NN Parameters for generation.")]

@app.command()
def train(model: Model,
          config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Train (or continue training of) a neural network
        flow model for generating structures.
    """
    pass

@app.command()
def generate(model: Model,
             n: Annotated[int, typer.Argument(help="Number of atoms per structure.")],
             s: Annotated[int, typer.Argument(help="Number of structures to generate.")],
             rho: Annotated[float, typer.Argument(help="Atomic density.")]):
    """ Use a trained model to generate structures.
    """
    pass

@app.command()
def dynamics(config: Annotated[Path, typer.Argument(help="DynamicSetup parameters.")],
             s: Annotated[int, typer.Argument(help="Number of structures to generate.")],
             out: Annotated[Path, typer.Argument(help="Output file.")],
             n: Annotated[int, typer.Argument(help="Number of atoms per structure.")],
             rho: Annotated[float, typer.Argument(help="Atomic density.")]):
    """ Run dynamics on a model to generate structures.
    """

    config = DynamicSetup.model_validate(**load_dict(config))
    data = LJDataset(config, s, out)
    data.process()
