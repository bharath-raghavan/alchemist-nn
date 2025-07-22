from typing import Dict, Optional, List
from pydantic import BaseModel, model_validator
import importlib, yaml, json
import torch
from ..data import transforms
from ..nn.embedding.default import Default
from ..utils.conversion import kelvin_to_lj, time_to_lj, dist_to_lj
from ..nn.flow.loss import Alchemical_NLL
from ..nn.flow.model import FlowModel
from ..nn.flow.network import NetworkWrapper
from ..nn.node.scalar import ScalarNodeModel
from ..nn.node.egnn import EGNN

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

class NetworkModelParams(BaseModel):
    def get(self):
        ret = self.dict()
        ret_keys = list(ret.keys())
        for i in ret_keys:
            if ret[i] is None or i == 'type':
                ret.pop(i)
        return ret
        
class EnergyModelParams(NetworkModelParams):
    type: str
    
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
    
    ## EGNN
    hidden_nf: Optional[int] = None
    act_fn: Optional[str] = None
    coords_weight: Optional[int] = None
    attention : Optional[bool] = None
    clamp: Optional[bool] = None
    norm_diff : Optional[bool] = None
    tanh : Optional[bool] = None
    ##

class EGNNParams(NetworkModelParams):
    hidden_nf: Optional[int] = 128
    n_layers: Optional[int] = 3

class EGNNParams(NetworkModelParams):
    hidden_nf: Optional[int] = 128
    n_layers: Optional[int] = 3
                
class FlowParams(UnittedParams):
    dt: float = 1
    n_iter: int
    scalar_hidden_nf: int
    energy: EnergyModelParams
    egnn: EGNNParams
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
        if self.energy.cutoff != None:
            self.energy.cutoff = dist_to_lj(self.energy.cutoff, unit=self.units.dist)
        if self.energy.atomref != None:
            self.energy.atomref = list(self.energy.atomref)
        self.box = [dist_to_lj(float(i), unit=self.units.dist) for i in self.box]
        return self
            
    def get(self, atom_types):
        networks = []
        
        if self.prec == 64:
            dtype = torch.float64
        else:
            dtype = 32
        
        for i in range(self.n_iter):
            energy_model_class = getattr(importlib.import_module(f"alchemist.nn.energy.{self.energy.type}"), f"{self.energy.type.upper()}")
            energy_network = energy_model_class(self.node_nf, self.box, **self.energy.get())
            node_network = ScalarNodeModel(self.scalar_hidden_nf, 1, self.scalar_hidden_nf)
            node_force_network = EGNN(self.scalar_hidden_nf, self.egnn.hidden_nf, self.egnn.n_layers)
            networks.append(NetworkWrapper(energy_network, node_network, node_force_network))
                
        return FlowModel(networks, Default(dtype, self.node_nf, self.scalar_hidden_nf), time_to_lj(self.dt, unit=self.units.time), self.box, dtype)

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

