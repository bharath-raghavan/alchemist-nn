from typing import Dict, Optional, List
from pydantic import BaseModel, model_validator
import importlib
import yaml
import json
from .data import transforms
from .nn.egcl import EGCL
from .nn.argmax import ArgMax
from .utils.conversion import kelvin_to_lj, time_to_lj
from .flow.loss import Alchemical_NLL

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
        
    def get(self):
        dataset_class = getattr(importlib.import_module(f"alchemist.data.{self.type}"), f"{self.type.upper()}Dataset")
        
        dataset_args = self.params
        T = [transforms.ConvertPositionsFrom(dataset_args['units']['dist']), transforms.Center()]

        if 'randomize_vel' in dataset_args and dataset_args['randomize_vel']:
            T.append(transforms.RandomizeVelocity(kelvin_to_lj(float(dataset_args['temp']))))
            dataset_args.pop('temp')
        else:
            T.append(transforms.ConvertVelocitiesFrom(dataset_args['units']['dist'], dataset_args['units']['time']))
        
        dataset_args['dist_unit'] = dataset_args['units']['dist']
        dataset_args['time_unit'] = dataset_args['units']['time']
        dataset_args.pop('units')
        
        return dataset_class(**dataset_args, transform=transforms.Compose(T))

class NetworkParams(BaseModel):
    type: str = 'egcl'
    n_iter: int
    hidden_nf: int
    
    def get(self, node_nf):
        networks = []
        for i in range(self.n_iter): networks.append(EGCL(node_nf, node_nf, self.hidden_nf))
        return networks

class FlowParams(UnittedParams):
    type: str = 'lf'
    dt: float = 1
    network: NetworkParams
    checkpoint: Optional[str] = None
    
    def get(self, node_nf):
        integrator_class = getattr(importlib.import_module("alchemist.flow.dynamics"), f"{self.type.upper()}Integrator")
        return integrator_class(self.network.get(node_nf), ArgMax(node_nf, self.network.hidden_nf), time_to_lj(self.dt, unit=self.units.time))

class LossParams(BaseModel):
    temp: float
    softening: float
    
    def get(self):
        return Alchemical_NLL(kBT=kelvin_to_lj(self.temp), softening=self.softening)

class TrainingParams(BaseModel):
    num_epochs: int
    lr: float
    scheduler_type: Optional[str] = None
    scheduler_params: Optional[Dict] = None
    loss: LossParams
    log_interval: int
    batch_size: int = 100
    accum_iter: int = 0

class ConfigFile(BaseModel):
    checkpoint: Optional[str] = None
    flow: Optional[FlowParams] = None
    training:  Optional[TrainingParams] = None
    dataset:  Optional[DatasetParams] = None
    generate:  Optional[DatasetParams] = None
    
    @staticmethod
    def fromFile(input):
        with open(input, "r", encoding="utf-8") as f:
            if input.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    
        return ConfigFile(**data)
            
    @model_validator(mode='before')
    def _check_whether_units_present(cls, values):
        for key in values:
            if key in ['flow', 'dataset', 'generate']:
                if 'units' not in values[key]:
                    values[key]['units'] = values['units']
                if key == 'generate':
                    values[key]['type'] = 'lj'
                    values[key]['random_h'] = True # do not do one-hot encoding for h, make gaussian instead
                    if 'temp' not in values[key]:
                        values[key]['temp'] = values['training']['loss']['temp']
                    if 'softening' not in values[key]:
                        values[key]['softening'] = values['training']['loss']['softening'] 
                    if 'box' not in values[key]:
                        values[key]['box'] = values['dataset']['box'] 
                    if 'atom_types' not in values[key]:
                        values[key]['atom_types'] = values['dataset']['atom_types'] 
        return values

