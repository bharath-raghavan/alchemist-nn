from sys import stdout
from abc import ABC, abstractmethod
import math
import numpy as np
import torch
from .base import InMemoryBaseDataset, Data
from ..utils.helpers import apply_pbc, one_hot
from ..utils.constants import atom_types
from ..utils.conversion import kelvin_to_lj

import openmm as mm
import openmm.app as app
import openmm.unit as unit

class SimulatedDatasetReporter:
    def __init__(self, random_h, r_cut, transform, report_interval, report_from, desc, dist_units, time_units, traj, temp, atom_types):
        self.data_list = []
        self.transform = transform
        self.random_h = random_h
        self.r_cut = r_cut
        self.box_pad = 0
        self.report_interval = report_interval
        self.report_from = report_from
        self.desc = desc
        self.dist_units = dist_units
        self.time_units = time_units
        self.traj = traj
        self.atom_types = atom_types

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
        
        if self.random_h:
            h = torch.normal(0, 1, size=(N, len(self.atom_types)), dtype=torch.float64)
        else:
            atom_types = {z:i for i,z in enumerate(self.atom_types)}
            type_idx = [atom_types[i] for i in z]
            h = one_hot(torch.tensor(type_idx), num_classes=len(atom_types), dtype=self.prec)
        
        g = torch.normal(0, 1, size=h.shape, dtype=torch.float64)
        
        data = Data(
            z=z,
            h=h,
            g=g,
            pos=pos,
            vel=torch.tensor(state.getVelocities().value_in_unit(self.dist_units/self.time_units), dtype=torch.float64),
            N=N,
            box=box.repeat(N, 1),
            r_cut=self.r_cut,
            label=f'Simulated dataset: {self.desc} Frame: {simulation.currentStep}'
            )
        
        self.data_list.append(self.transform(data))

class SimulatedDataset(InMemoryBaseDataset, ABC):
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
        
        
        self.random_h = False # should I tell SimulatedDatasetReporter to randomize h, this is set to true only in LJ Dataset
        
        self.integrator = mm.LangevinMiddleIntegrator(temp*unit.kelvin, friction/(scale*unit.picosecond), dt*scale*unit.picoseconds)
        simulation, desc = self.setup(**input_params)
        
        print("Running minimization")
        simulation.minimizeEnergy()
        
        simulation.context.setVelocitiesToTemperature(temp*unit.kelvin)
        
        if 'atom_types' in input_params:
            atom_types = input_params['atom_types']
        else:
            atoms_types = ['X'] * int(input_params['natom_types'])
        
        # Add reporters to get data and output traj
        rep = SimulatedDatasetReporter(self.random_h, self.r_cut, self.transform, report_interval, report_from, desc, self.dist_units, time_units, traj, temp, atoms_types)
        simulation.reporters.append(rep)
        
        # Add reporters to output log
        simulation.reporters.append(app.StateDataReporter(log, report_interval, step=True, potentialEnergy=True, temperature=True))
        simulation.reporters.append(app.StateDataReporter(stdout, report_interval, step=True, potentialEnergy=True, temperature=True))
        
        print("Running MD simulation")
        simulation.step(n_iter)
        self.data_list = rep.data_list # capture data list from rep