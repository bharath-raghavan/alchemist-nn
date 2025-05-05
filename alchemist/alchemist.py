from pathlib import Path
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import logging
_logger = logging.getLogger(__name__)

import typer
app = typer.Typer()

from .config import load_config

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
def dynamics(model: Model,
             n: Annotated[int, typer.Argument(help="Number of atoms per structure.")],
             s: Annotated[int, typer.Argument(help="Number of structures to generate.")],
             rho: Annotated[float, typer.Argument(help="Atomic density.")]):
    """ Use a trained model to generate structures.
    """
    pass
