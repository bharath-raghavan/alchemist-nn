from typing import List, Optional
from pathlib import Path
import json

from pydantic import BaseModel
import yaml

class Units(BaseModel):
    time: str
    dist: str

class TrajectoryData(BaseModel):
    batch_size: int = 100
    top_file: List[str]
    traj_file: List[str]
    processed_file: str
    units: Units

class NetworkSetup(BaseModel):
    hidden_nf: int

"""
class DynamicSetup(BaseModel):
    cls: str = "LJDataset"
    r_cut: float
    report_interval: int = 10
    report_from: int = 100
    temp: float = 300
    traj_file: str
    dt: float
    integrator: str
    n_iter: int
    #box_pad: float = 1
    #network: NetworkSetup
"""

class DynamicConfig(BaseModel):
    nparticles: int = 216
    substeps: int = 100

    reduced_density: float = 0.85 # reduced density rho*sigma^3
    temperature: float = 120 # Kelvin
    collision_rate: int = 5 # 1/ps
    timestep: float = 2.5 # fs
    sigma: float = 3.4 # angstrom
    pressure: Optional[float] = None # bar | None

class LossSetup(BaseModel):
    temp: float = 298.15
    softening: float

class TrainingSetup(BaseModel):
    num_epochs: int
    lr: float
    scheduler: bool
    loss: LossSetup
    log_interval: int

#class ConfigFile(BaseModel):
#    dynamics: DynamicSetup
#    training: TrainingSetup
#    dataset: TrajectoryData

def load_dict(fname: Path) -> dict:
    """ Read a dict from a yaml or json-formatted file.
    """
    with open(fname, "r", encoding="utf-8") as f:
        if fname.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data

#def load_config(fname: Path) -> ConfigFile:
#    """ Read a ConfigFile from a yaml or json-formatted
#    file.
#    """
#    return ConfigFile.model_validate(**load_dict(fname))
