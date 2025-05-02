import os
import sys
from alchemist.main import Main
        
if __name__ == "__main__":
    main_hndl = Main(world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'), local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
    main_hndl(sys.argv[1]) 
