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

def mk_system(nparticles, mass, sigma, epsilon, box_edge):
    system = System()

    system.setDefaultPeriodicBoxVectors(Vec3(box_edge, 0, 0),
                                        Vec3(0, box_edge, 0),
                                        Vec3(0, 0, box_edge))
    force = CustomNonbondedForce("""4*epsilon*(1/((alphaLJ + (r/sigma)^6)^2) - 1/(alphaLJ + (r/sigma)^6));
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

def run(system, integrator, positions):
    context = Context(system, integrator)

    # Initiate from last set of positions.
    context.setPositions(positions)

    # Minimize energy from coordinates.
    print("minimizing...")
    LocalEnergyMinimizer.minimize(context)

    # Equilibrate.
    print("equilibrating...")
    integrator.step(nequil_steps)

    # Sample.
    position_history = [] # position_history[i] is the set of positions after iteration i
    for iteration in range(nprod_iterations):
        print("production iteration %d/%d" % (iteration+1, nprod_iterations))

        # Run dynamics.
        integrator.step(nprod_steps)

        # Get coordinates.
        state = context.getState(positions=True)
        positions = state.getPositions(asNumpy=True)

        # Store positions.
        position_history.append(positions)

        #context.setPositions(fluid.positions)
        #integrator.step(n_steps)
        #print(context.positions)

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

# Simulation settings
#pressure = 80*unit.atmospheres
pressure = None
reduced_density = 0.85 # reduced density rho*sigma^3

temperature = 120*unit.kelvin
collision_rate = 5/unit.picoseconds
timestep = 2.5*unit.femtoseconds

nparticles = 216
nequil_steps = 100
nprod_iterations = 10
nprod_steps = 100

# Create a Lennard Jones test fluid mimicking Argon
mass = 39.9 * unit.amu
sigma = 3.4*unit.angstrom
epsilon = 0.238 * unit.kilocalories_per_mole
charge = 0.0 * unit.elementary_charge


# =============================================================================
# Compute box size.
# =============================================================================

volume = nparticles*(sigma**3)/reduced_density
box_edge = volume**(1.0/3.0)
cutoff = min(box_edge*0.49, 2.5*sigma) # Compute cutoff
print("sigma = %s" % sigma)
print("box_edge = %s" % box_edge)
print("cutoff = %s" % cutoff)

system = mk_system(nparticles, mass, sigma, epsilon, box_edge)

# random initial positions
import numpy.random
positions = unit.Quantity(
                numpy.random.uniform(high=box_edge/unit.angstroms,
                size=[nparticles,3]), unit.angstrom) #sic


integrator = LangevinIntegrator(temperature, collision_rate, timestep)

history = run(system, integrator, positions)
for x in history:
    print(x[0])

en = compute_energies(system, history, temperature)
print(en)
