from rdkit import Chem
M = Chem.GetPeriodicTable().GetAtomicWeight('Ar') # in amu
sigma = 3.4e-10 # m
eps = 0.238e3 # J/mol
kB = 8.3144621 # J/(K mol)

atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
