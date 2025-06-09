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
def dynamics(config: Annotated[Path, typer.Argument(help="DynamicConfig parameters.")],
             s: Annotated[int, typer.Argument(help="Number of structures to generate.")],
             out: Annotated[Path, typer.Argument(help="Output file.")]
            ):
    """ Run dynamics on a model to generate structures.
    """

    #config = DynamicConfig.model_validate(**load_dict(config))
    config = DynamicConfig.model_validate(load_dict(config))
    #data = LJDataset(config, s, out)
    #data.process()

    volume = config.nparticles*(config.sigma**3)/config.reduced_density
    box_edge = volume**(1.0/3.0)

    names = ['Ar']*config.nparticles
    cell  = np.eye(3)*box_edge
    def get_ase(x, e):
        return to_ase(names, x, cell, energy=e)

    async def map_fn(it, fn):
        i = 0
        for z in it:
            yield fn(*z)
            i += 1
            if i >= s:
                break

    def c2(inp, out, *args):
        return out(inp, *args)

    async def consume(stream):
        async for i in stream:
            pass

    asyncio.run(
            consume( add_mols(map_fn(simulate_system(config), get_ase), out) )
        )
