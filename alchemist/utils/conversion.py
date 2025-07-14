import math

from .constants import *

def meter_to_lj(x):
    return x/sigma
    
def meter_per_sec_to_lj(x):
    return x*math.sqrt(M/eps)

def amu_to_lj(m):
    return m/M
    
def second_to_lj(t):
    return t*math.sqrt(eps/M)/sigma
    
def time_to_lj(t, unit='pico'):
    if unit == 'pico': a = 1e-12
    elif unit == 'femto': a = 1e-15
    return second_to_lj(t*a)
    
def dist_to_lj(x, unit='ang'):
    if unit == 'ang': a = 1e-10
    elif unit == 'nm': a = 1e-9
    return meter_to_lj(x*a)

def vel_to_lj(x, unit1='ang', unit2='pico'):
    if unit1 == 'ang':
        a = 1e-10
    elif unit1 == 'nm':
        a = 1e-9
    if unit2 == 'pico':
        b = 1e-12
    elif unit2 == 'femto':
        b = 1e-12
    return meter_per_sec_to_lj(x*a/b)

def kelvin_to_lj(T):
    return T * kB/eps
    
def lj_to_kelvin(kBT):
    return kBT*eps/kB

def lj_to_meter(x_):
    return x_*sigma
    
def lj_to_meter_per_sec(x):
    return x*math.sqrt(eps/M)
    
def lj_to_dist(x_, unit='ang'):
    if unit == 'ang': a = 1e-10
    elif unit == 'nm': a = 1e-9
    return lj_to_meter(x_/a)
    
def lj_to_vel(x_, unit1='ang', unit2='pico'):
    if unit1 == 'ang': a = 1e-10
    elif unit1 == 'nm': a = 1e-9
    if unit2 == 'pico':
        b = 1e-12
    elif unit2 == 'femto':
        b = 1e-12
    return lj_to_meter_per_sec(x_*b/a)
