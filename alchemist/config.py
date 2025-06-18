from typing import Dict Optional
from pydantic import BaseModel

class UnitsParams(BaseModel):
    time: str
    dist: str

class UnittedParams(BaseModel):
    units: UnitsParams

class DatasetParams(UnittedParams):
    type: str
    batch_size: int = 1
    params: Dict

    @model_validator(mode='before')
    def _dump_params(cls, values):
        values['params'] = {}
        for key in values:
            if key not in ['type', 'batch_size', 'params']:
                values['params'][key] = values[key]
        return values

class NetworkParams(BaseModel):
    type: str = 'egcl'
    n_iter: int

class FlowParams(UnittedParams):
    type: str = 'lf'
    dt: int = 1
    r_cut: float = 3
    box: List
    network: NetworkParams
    checkpoint: Optional[str] = None

class LossParams(BaseModel):
    temp: float # TODO: convert to LJ here only
    softening: float

class TrainingParams(BaseModel):
    num_epochs: int
    lr: float
    scheduler_type: Optional[str] = None
    scheduler_params: Optional[Dict] = None
    loss: LossParams
    log_interval: int
    batch_size: int = 100

class ConfigFile(BaseModel):
    flow: FlowParams
    training:  Optional[TrainingParams] = None
    dataset:  Optional[DatasetParams] = None
    generate:  Optional[DatasetParams] = None
    
    @model_validator(mode='before')
    def _check_whether_units_present(cls, values):
        for key in values:
            if key in ['flow', 'dataset', 'generate']:
                if 'units' not in values[key]:
                    values[key]['units'] = values['units']
                if key == 'generate':
                    values[key]['type'] = 'lj'
                    if 'temp' not in values[key]:
                        values[key]['temp'] = values['training']['loss']['temp']
                    if 'softening' not in values[key]:
                        values[key]['softening'] = values['training']['loss']['softening'] 
        return values

