# AlchemistNN

A package for generating chemical structures
using artificial neural networks.

Run by passing a YAML file:

alchemist train train.yaml

or on slurm (will use DDP to run in parallel):

srun alchemist train train.yaml
