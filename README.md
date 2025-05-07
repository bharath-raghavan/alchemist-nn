# AlchemistNN

A package for generating chemical structures
using artificial neural networks.

Run by passing a YAML file:

python __main__.py config.yaml

or on slurm (will use DDP to run in parallel):

srun python __main__.py config.yaml

## YAML File

Here is an example of a YAML file that will train on an MDDataset:

```
mode: train
units:
  time: pico
  dist: ang
dataset:
  type: md
  batch_size: 1000
  top_file: [data/acetone/raw/acetone.tpr]
  traj_file: [data/acetone/raw/acetone.trr]
  processed_file: data/acetone/processed.pt
dynamics:
  integrator: lf
  n_iter: 5
  dt: 1
  checkpoint_path: model.cpt
  network:
    hidden_nf: 128
training:
  num_epochs: 5000
  batch_size: 5
  lr: 1e-3
  scheduler: No
  scheduler_step: 1
  gamma: 0.99
  loss:
    temp: 300
    softening: 0.1
  log_interval: 1
```

This has the following sections:

```
mode: train
```

Set the mode here: other options are `generate` (which will run the `reverse()` on the model), and `dataset` (which will just read the `dataset` section, write out the processed file and exit).

```
units:
  time: pico
  dist: ang
```

Set the units of the dataset, and all other parameters given in the input file. Other options include `nm` and 'femto' for distance and time respectively. Note that MDAnalysis always processes datasets in Angstroms and picoseconds, regardless of what the units of the trajectories and topologies are. So, if you are using an MDDataset, always set the units of `pico` and `ang`. If using a `SimulatedDataset` child class (which uses OpenMM), any units can be used.

```
dataset:
  type: md
  batch_size: 1000
  top_file: [data/acetone/raw/acetone.tpr]
  traj_file: [data/acetone/raw/acetone.trr]
  processed_file: data/acetone/processed.pt
```

Sets the dataset to be used. Type are: `md`, `sdf`, `lig`, `lj`, `compose`. The first two read from files, and the last two will run an OpenMM simulation.

Let us take a look at the `lj` dataset:

```
dataset:
  type: lj
  batch_size: 1
  discard: -1
  box: [20, 20, 20]
  n_atoms: 735
  n_iter: 10000
  interval: 100
  log: data/lj/log.txt
  traj: data/lj/traj.pdb
  processed_file: data/lj/processed.pt
  temp: 300
  friction: 1
  dt: 0.004
  cutoff: 1
  r_cut: 3
```

This will arrange 735 Argon atoms in a box of 20x20x20 (in the units specified). Then run NVT MD for 10000 at 300K with the friction and dt as given. The `cutoff` is used while running the MD (it is a multiple of `sigma`). It will print out (both to `stdout` and the log file) the temperature and energy every 100 steps. Starting from the iteration specified in discard, it will write the trajectory to a PDB file (specified in `traj`) and to `processed_file` (this will be used to load the dataset for training/generation). Here the `discard` is -1, so it will store only the last frame of the simulation. The `r_cut` is used to calculate the neighbor list during the NN training.

Let us take a look at the `lig` dataset:

```
dataset:
  type: lig
  name: benzene
  smiles: c1ccccc1
  discard: 2000
  padding: 10
  r_cut: 3
  temp: 300
  force_field: [amber/protein.ff14SB.xml, amber/tip3p_standard.xml, amber/tip3p_HFE_multivalent.xml]
  n_iter: 50000
  interval: 500
  log: data/benzene/log.txt
  traj: data/benzene/traj.pdb
  processed_file: data/benzene/processed.pt
  friction: 1
  dt: 0.004
```

Here you give it a `name` (just for documentation/debugging purposes) and a smiles string. A 3D configuration from the simles string is generated, and a box is created 10 units (units specified in the `units` section) greater from the minimum bounding box of the molecule as specified in `padding`. The box is solvated with waters, and NVT MD is run at 300K for 50000 iterations with the forcefields as specified. The rest of the options are same as above.

Finally, you can also string multiple datasets together with `compose`:

```
dataset:
  type: compose
  number: 3
dataset1:
  type: lig
  processed_file: /ccs/proj/stf006/raghavan/enflow/test/data/benzene/processed.pt
dataset2:
  type: md
  processed_file: /ccs/proj/stf006/raghavan/enflow/test/data/acetone/processed.pt
dataset3:
  type: lig
  processed_file: /ccs/proj/stf006/raghavan/enflow/test/data/vanillin/processed.pt
```

Each dataset could have their own options to process them as described above, or you can just pass the processed file.

The `dynamics` section sets the flow:

```
dynamics:
  integrator: lf
  n_iter: 5
  dt: 1
  checkpoint_path: model.cpt
  network:
    hidden_nf: 128
```

Integrator is the flow coupling, it can be `lf` (LeapFrog) or `vv` (VelocityVerlet). I velocity verlet is not necessary, so we can remove it. The rest of the options are self explanatory. I also have a `network` option that sets the network parameters. Currently, only the EGNN is implemented, but I would like to eventually like to add other networks like Nequip. We might get better results.

The last section is on the `training`, not much to be explained there.
