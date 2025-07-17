import torch

from ..utils.conversion import dist_to_lj, vel_to_lj, kelvin_to_lj

import numpy as np
from rdkit import Chem
from scipy.special import erf
from scipy.interpolate import interp1d as interp

class NoneTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        return data

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class ConvertPositionsFrom:
    def __init__(self, input_unit):
        self.input_unit = input_unit

    def __call__(self, data):
        data.pos = dist_to_lj(data.pos, self.input_unit)
        return data
        
class ConvertVelocitiesFrom:
    def __init__(self, input_unit1, input_unit2):
        self.input_unit1 = input_unit1
        self.input_unit2 = input_unit2

    def __call__(self, data):
        data.vel = vel_to_lj(data.vel, self.input_unit1, self.input_unit2)
        return data
        
class Center:
    def __init__(self):
        pass

    def __call__(self, data):
        pos = data.pos
        data.pos = pos - pos.mean(dim=0, keepdim=True)
        return data

class RandomizeVelocity:
    def __init__(self, kBT):
        self.kBT = kBT
        self.pse = Chem.GetPeriodicTable()
        
    def generate_velocities(self, n):
        """ generate a set of velocity vectors in 3D from the MB inverse CDF function """
        """ Adapted from: https://github.com/tpogden/quantum-python-lectures/blob/master/11_Monte-Carlo-Maxwell-Boltzmann-Distributions.ipynb """
        
        # Cumulative Distribution function of the Maxwell-Boltzmann speed distribution
        m = 1 # TODO: change mass to that of each atom
        a = np.sqrt(self.kBT/m)
        MB_CDF = lambda v : erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a
        v = np.arange(0,2500,0.1)
        cdf = MB_CDF(v)

        #create interpolation function to CDF
        inv_cdf = interp(cdf,v)
                                 
        rand_nums = np.random.random(n)
        speeds = inv_cdf(rand_nums)
    
        theta = np.arccos(np.random.uniform(-1,1,n))
        phi = np.random.uniform(0,2*np.pi,n)
    
        # convert to cartesian units
        vx = speeds * np.sin(theta) * np.cos(phi) 
        vy = speeds * np.sin(theta) * np.sin(phi)
        vz = speeds * np.cos(theta)
    
        return np.stack([vx, vy, vz], axis=1, dtype=np.float64)

    def __call__(self, data):
        data.vel = torch.from_numpy(self.generate_velocities(data.N))
        return data
        

