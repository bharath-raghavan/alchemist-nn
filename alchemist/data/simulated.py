from sys import stdout
from abc import ABC, abstractmethod
import math
import numpy as np
import torch
from .base import BaseDataset, Data
from ..utils.helpers import apply_pbc, one_hot
from ..utils.constants import atom_types
from ..utils.conversion import kelvin_to_lj

import openmm as mm
import openmm.app as app
import openmm.unit as unit

class SimulatedDatasetReporter(BaseDataset):
    def __init__(self, node_nf_input, r_cut, transform, report_interval, report_from, desc, dist_units, time_units, traj, temp):
        self.data_list = []
        self.transform = transform
        self.node_nf_input = node_nf_input
        self.r_cut = r_cut
        self.box_pad = 0
        self.report_interval = report_interval
        self.report_from = report_from
        self.desc = desc
        self.dist_units = dist_units
        self.time_units = time_units
        self.traj = traj
        self.kBT = kelvin_to_lj(temp)

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep%self.report_interval
        return {'steps': steps, 'periodic': False, 'include':['positions', 'velocities']} # OpenMM's PBC application is not great, we will do it ourselves
    
    def process(self, **input_params):
        pass
        
    def report(self, simulation, state):
        if simulation.currentStep < self.report_from: return
        
        pos = torch.tensor(state.getPositions().value_in_unit(self.dist_units), dtype=torch.float64)
        N = pos.shape[0]
        
        box_vec3 = simulation.topology.getUnitCellDimensions().value_in_unit(self.dist_units)
        box = torch.tensor([box_vec3[0], box_vec3[1], box_vec3[2]], dtype=torch.float64)
        
        pos = apply_pbc(pos, box)
        
        # since we changed positions, we need to write the pdb ourselves
        with open(self.traj, 'a') as pdbfile:
            app.PDBFile.writeHeader(simulation.topology, pdbfile)
            pdbfile.write(f"MODEL        {simulation.currentStep}\n")
            app.PDBFile.writeModel(simulation.topology, pos, pdbfile)
            pdbfile.write("ENDMDL\n")
            
        z = [a.element.symbol for a in simulation.topology.atoms()]
        
        if self.node_nf_input:
            h = torch.normal(0, 1/math.sqrt(self.kBT), size=(N, self.node_nf_input), dtype=torch.float64)
        else:
            h = None
        
        self.append(
            z=z,
            h=h,
            pos=pos,
            vel=torch.tensor(state.getVelocities().value_in_unit(self.dist_units/self.time_units), dtype=torch.float64),
            N=N,
            box=box,
            label=f'Simulated dataset: {self.desc} Frame: {simulation.currentStep}'
        )

class SimulatedDataset(BaseDataset, ABC):
    @abstractmethod
    def setup(self, **input_params):
        pass
        
    def process(self, **input_params):
        temp = input_params['temp']
        report_interval = input_params['interval']
        report_from = input_params['discard']
        if report_from == -1: report_from = report_interval
        log = input_params['log']
        traj = input_params['traj']
        n_iter = input_params['n_iter']
        dist_units = input_params['dist_unit']
        time_units = input_params['time_unit']
        dt = input_params['dt']
        friction = input_params['friction']
        
        if dist_units == 'ang':
            self.dist_units = unit.angstrom
        elif dist_units == 'nm':
            self.dist_units = unit.nanometers
    
        scale = 1
        if time_units == 'pico':
            time_units = unit.picoseconds
        elif time_units == 'femto':
            time_units = unit.femtoseconds
            scale = 1e-3
        
        self.random_h = False
        
        self.integrator = mm.LangevinMiddleIntegrator(temp*unit.kelvin, friction/(scale*unit.picosecond), dt*scale*unit.picoseconds)
        simulation, desc = self.setup(**input_params)
        
        print("Running minimization")
        simulation.minimizeEnergy()
        
        simulation.context.setVelocitiesToTemperature(temp*unit.kelvin)
        
        if self.random_h: # set by LJ dataset
            node_nf_input = input_params['node_nf']
        else:
            node_nf_input = None
        
        # Add reporters to get data and output traj
        rep = SimulatedDatasetReporter(node_nf_input, self.r_cut, self.transform, report_interval, report_from, desc, self.dist_units, time_units, traj, temp)
        simulation.reporters.append(rep)
        
        # Add reporters to output log
        simulation.reporters.append(app.StateDataReporter(log, report_interval, step=True, potentialEnergy=True, temperature=True))
        simulation.reporters.append(app.StateDataReporter(stdout, report_interval, step=True, potentialEnergy=True, temperature=True))
        
        print("Running MD simulation")
        simulation.step(n_iter)
        self.data_list = rep.data_list # capture data list from rep