# Use OpenMM to create a trajectory of LJ liquid
#

import openmm
from openmm import (
    unit,
    Context,
    CustomNonbondedForce,
    NonbondedForce,
    LangevinIntegrator,
    VerletIntegrator,
    LocalEnergyMinimizer,
    System,
    Vec3,
)
import numpy as np

from .config import DynamicConfig

def mk_system(nparticles, mass, sigma, epsilon, box_edge, cutoff, pressure, temperature):
    system = System()

    system.setDefaultPeriodicBoxVectors(Vec3(box_edge, 0, 0),
                                        Vec3(0, box_edge, 0),
                                        Vec3(0, 0, box_edge))
    eta = (cutoff/unit.angstrom) / 3.0 # Note: eta*cutoff = 3, erfc(3) = 2.2e-5
    force = CustomNonbondedForce(f"""4*epsilon*(1/((alphaLJ + (r/sigma)^6)^2) - 1/(alphaLJ + (r/sigma)^6)) + charge1*charge2*erfc(r*{eta})/(alphaLJ+r);
                                    sigma=0.5*(sigma1+sigma2);
                                    epsilon=sqrt(epsilon1*epsilon2);
                                    alphaLJ=0.0;""")
    #           l12=1-(1-lambda)*step(useLambda1+useLambda2-0.5)""")
    force.addPerParticleParameter("sigma")
    force.addPerParticleParameter("epsilon")
    force.addPerParticleParameter("charge")
    #force.addPerParticleParameter("useLambda")

    #force.addGlobalParameter("lambda", lambda_value)
    force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)
    for particle_index in range(nparticles):
        system.addParticle(mass)
        #if particle_index == 0:
        #   # Add alchemically-modified particle.
        #   force.addParticle([sigma, epsilon, 1])
        #else:
        # Add normal particle.
        force.addParticle([sigma, epsilon, 0])
    system.addForce(force)

    # Add a barostat
    if pressure is not None:
        assert temperature is not None
        barostat = MonteCarloBarostat(pressure, temperature)
        system.addForce(barostat)
    return system

# Generator that yields successive sampled frames
def run(system, integrator, positions, substeps):
    context = Context(system, integrator)
    T = integrator.getTemperature()
    #print(f"temp = {T}")
    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * T)

    # Initiate from last set of positions.
    context.setPositions(positions)

    # Minimize energy from coordinates.
    print("minimizing...")
    LocalEnergyMinimizer.minimize(context)

    # Equilibrate.
    #print("equilibrating...")
    #integrator.step(nequil_steps)

    # Sample.
    while True:
        # Run dynamics.
        integrator.step(substeps)

        # Get coordinates.
        state = context.getState(positions=True, energy=True)
        positions = state.getPositions(asNumpy=True)
        energy = beta * state.getPotentialEnergy()
        yield positions/unit.angstrom, energy

    # Clean up.
    del context, integrator
    return position_history

def compute_energies(system, coords, temperature):
    print("computing energies at all states...")
    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature) # inverse temperature

    # Set up Context just to evaluate energies.
    integrator = VerletIntegrator(1*unit.femtoseconds)
    context = Context(system, integrator)

    # Compute reduced potentials of all snapshots.
    energies = []
    for x in coords:
        context.setPositions(x)
        state = context.getState(energy=True)
        energies.append( beta * state.getPotentialEnergy() )

    # Clean up.
    del context, integrator
    return energies

def simulate_system(config: DynamicConfig):
    # Simulation settings
    pressure = config.pressure*unit.bar if config.pressure else None
    temperature = config.temperature*unit.kelvin

    nparticles = config.nparticles

    # Create a Lennard Jones test fluid mimicking Argon
    mass = 39.9 * unit.amu
    sigma = config.sigma * unit.angstrom
    epsilon = 0.238 * unit.kilocalories_per_mole
    charge = 0.0 * unit.elementary_charge

    # =============================================================================
    # Compute box size.
    # =============================================================================

    volume = nparticles*(sigma**3)/config.reduced_density
    box_edge = volume**(1.0/3.0)
    cutoff = min(box_edge*0.49, 2.5*sigma) # Compute cutoff
    #print("sigma = %s" % sigma)
    #print("box_edge = %s" % box_edge)
    #print("cutoff = %s" % cutoff)

    system = mk_system(nparticles, mass, sigma, epsilon, box_edge, cutoff, pressure, temperature)

    # random initial positions
    #positions = unit.Quantity(
    #                np.random.uniform(high=box_edge/unit.angstrom,
    #                size=[nparticles,3]), unit.angstrom) #sic
    positions = np.random.uniform(high=box_edge/unit.angstrom,
                                  size=[nparticles,3]) * unit.angstrom

    integrator = LangevinIntegrator(
                            temperature,
                            config.collision_rate/unit.picoseconds,
                            config.timestep*unit.femtoseconds)

    yield from run(system, integrator, positions, config.substeps)
    #en = compute_energies(system, history, temperature)
    #print(en)

