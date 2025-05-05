from typing import List
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

class DynamicSetup(BaseModel):
    integrator: str
    n_iter: int
    dt: float
    r_cut: float
    box_pad: float = 1
    checkpoint_path: str
    network: NetworkSetup

class LossSetup(BaseModel):
    temp: float = 298.15
    softening: float

class TrainingSetup(BaseModel):
    num_epochs: int
    lr: float
    scheduler: bool
    loss: LossSetup
    log_interval: int

class ConfigFile(BaseModel):
    dynamics: DynamicSetup
    training: TrainingSetup
    dataset: TrajectoryData

def load_config(fname: Path) -> ConfigFile:
    """ Read a ConfigFile from a yaml or json-formatted
    file.
    """
    with open(fname, "r", encoding="utf-8") as f:
        if fname.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return ConfigFile.model_validate(**data)
