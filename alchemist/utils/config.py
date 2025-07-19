from typing import Dict, Optional, List
from pydantic import BaseModel, model_validator
import importlib, yaml, json
from ..data import transforms
from ..nn.argmax import ArgMax
from ..utils.conversion import kelvin_to_lj, time_to_lj, dist_to_lj
from ..nn.loss import Alchemical_NLL
from ..nn.flow import LFFlow
from ..nn.node import NodeModel

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

class EnergyModelParams(BaseModel):
    ## ViSNet
    lmax: Optional[int] = None
    vecnorm_type: Optional[str] = None
    trainable_vecnorm: Optional[bool] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    hidden_channels: Optional[int] = None
    num_rbf: Optional[int] = None
    trainable_rbf: Optional[bool] = None
    vertex: Optional[bool] = None
    atomref: Optional[List] = None
    reduce_op: Optional[str] = None
    cutoff: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    add_self_loops: Optional[bool] = None
    ##
    
    ## EGCL
    hidden_nf: Optional[int] = None
    act_fn: Optional[str] = None
    coords_weight: Optional[int] = None
    attention : Optional[bool] = None
    clamp: Optional[bool] = None
    norm_diff : Optional[bool] = None
    tanh : Optional[bool] = None
    ##
        
    def get(self):
        ret = self.dict()
        ret_keys = list(ret.keys())
        for i in ret_keys:
            if ret[i] is None:
                ret.pop(i)
        return ret
        
class FlowParams(UnittedParams):
    dt: float = 1
    n_iter: int
    node_nf: Optional[int] = None
    energy_model: str
    node_hidden_layers: int
    energy_model_params: EnergyModelParams
    box: List
    prec: int
    
    @model_validator(mode='before')
    def _check_whether_units_present(cls, values):
        for key in values:
            if key == 'energy_model_params':
                if 'units' not in values[key]:
                    values[key]['units'] = values['units']
        return values
    
    @model_validator(mode='after')
    def _convert(self):
        if self.energy_model_params.cutoff != None:
            self.energy_model_params.cutoff = dist_to_lj(self.energy_model_params.cutoff, unit=self.units.dist)
        #if self.energy_model_params.atomref != None :
        #    self.energy_model_params.atomref = list(self.energy_model_params.atomref)
        self.box = [dist_to_lj(float(i), unit=self.units.dist) for i in self.box]
        return self
            
    def get(self, node_nf):
        if self.node_nf is None:
            self.node_nf = node_nf
            
        energy_networks = []
        node_networks = []
        node_force_networks = []
        for i in range(self.n_iter):
            energy_model_class = getattr(importlib.import_module(f"alchemist.nn.energy.{self.energy_model}"), f"{self.energy_model.upper()}")
            energy_networks.append(energy_model_class(self.node_nf, self.box, **self.energy_model_params.get()))
            node_networks.append(NodeModel(self.node_nf, 1, self.node_hidden_layers))
            node_force_networks.append(NodeModel(self.node_nf, self.node_nf, self.node_hidden_layers))
                
        return LFFlow(energy_networks, node_networks, node_force_networks, ArgMax(self.node_nf, self.node_hidden_layers), time_to_lj(self.dt, unit=self.units.time), self.box, self.prec)

class LossParams(BaseModel):
    temp: Optional[float] = 300
    softening: Optional[float] = 0
    partition_func: Optional[float] = 10
    
    def get(self):
        return Alchemical_NLL(kBT=kelvin_to_lj(self.temp), partition_func=self.partition_func, softening=self.softening)

class TrainingParams(BaseModel):
    num_epochs: int
    lr: float
    scheduler_type: Optional[str] = None
    scheduler_params: Optional[Dict] = None
    loss: LossParams
    log_interval: int
    batch_size: int = 100
    accum_iter: int = 0

class ConfigParams(BaseModel):
    checkpoint: Optional[str] = None
    prec: Optional[int] = 64
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
    
        return ConfigParams(**data)
            
    @model_validator(mode='before')
    def _check_whether_units_present(cls, values):
        for key in values:
            if key in ['flow', 'dataset', 'generate']:
                values[key]['prec'] = values['prec']
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
                        values[key]['box'] = values['flow']['box'] 
                    if 'atom_types' not in values[key]:
                        values[key]['atom_types'] = values['dataset']['atom_types'] 
        return values

