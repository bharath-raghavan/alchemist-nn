from .simulated import SimulatedDataset

import openmm.app as app
import openmm.unit as unit
from openmm.vec3 import Vec3
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from openff.units.openmm import to_openmm
from openff.toolkit import Molecule, ForceField
from openff.interchange import Interchange
        
class LIGDataset(SimulatedDataset):       
    def setup(self, **input_params):
        smiles = input_params['smiles']
        if 'name' in input_params:
            lig_name = input_params['name']
        else:
            lig_name = 'ligand'
        
        ff = input_params['force_field']
        if 'n_conformers' in input_params:
            n_conformers = int(input_params['n_conformers'])
        else:
            n_conformers = 1
        
        if 'padding' in input_params:
            padding = float(input_params['padding'])
            box = None
        elif 'box' in input_params:
            box =  float(input_params['box'])
            padding = None
        else:
            print('error')
        
        molecule = Molecule.from_smiles(smiles)
        for atom in molecule.atoms:
            atom.metadata["residue_name"] = lig_name.upper()[:3]
            
        topology = molecule.to_topology().to_openmm()

        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
        ff = app.ForceField(*ff) # 
        ff.registerTemplateGenerator(smirnoff.generator)

        molecule.generate_conformers(n_conformers=n_conformers)
        positions = to_openmm(molecule.conformers[0]) # read more conformers

        modeller = app.Modeller(topology, positions)
        
        if padding:
            modeller.addSolvent(ff, padding=padding*self.dist_units)
        else:
            modeller.addSolvent(ff, boxSize=Vec3(*box)*self.dist_units)
            
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
        
        simulation = app.Simulation(modeller.topology, system, self.integrator)
        simulation.context.setPositions(modeller.positions)
        
        return simulation, f'Solvated {lig_name} ({smiles})'
