import os
import sys
import yaml
import json
import numpy as np
from datetime import timedelta
import time
import importlib

import torch

from .flow.loss import Alchemical_NLL
from .nn.egcl import EGCL
from .data.sdf import SDFDataset
from .data.base import DataLoader
from .data import transforms
from .utils.conversion import dist_to_lj, kelvin_to_lj, time_to_lj, lj_to_dist, lj_to_kelvin
from .utils.constants import sigma

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def write_xyz(out, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x in out.pos:
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % ('Ar', x[0].item(), x[1].item(), x[2].item()))
                
class Main:

    def __init__(self, input, world_size=None, world_rank=None, local_rank=None, num_cpus_per_task=None):
        if world_size and world_rank and local_rank:
            self.ddp = True
        else:
            self.ddp = False
            
        if self.ddp:
            self.world_size = int(world_size)
            self.world_rank = int(world_rank)
            self.local_rank = int(local_rank)

            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["RANK"] = str(self.world_rank)
            os.environ["LOCAL_RANK"] = str(self.local_rank)

            os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"
            torch.cuda.set_device(self.local_rank)
            device = torch.cuda.current_device()

            dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), init_method="env://", rank=self.world_rank, world_size=self.world_size)
    
            self.num_cpus_per_task = int(num_cpus_per_task)
            
            if world_rank == 0:
                eprint(f"Running DDP\nInitialized? {dist.is_initialized()}", flush=True)
        else:
            print("Running serially", flush=True)
            self.world_rank = 0
            self.local_rank = 'cpu'
    
        with open(fname, "r", encoding="utf-8") as f:
            if fname.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        self.config_file = ConfigParams(**data)        
    
    def _setup_dataset(dataset_config):
        dataset_args = dataset_config.params
        dataset_type = dataset_config.type
        dataset_class = getattr(importlib.import_module(f"alchemist.data.{dataset_type}"), f"{dataset_type.upper()}Dataset")
        
        T = [transforms.ConvertPositionsFrom(args['units']['dist']), transforms.Center()]

        if 'randomize_vel' in dataset_args and dataset_args['randomize_vel']:
            T.append(transforms.RandomizeVelocity(kelvin_to_lj(float(dataset_args['temp']))))
            dataset_args.pop('temp')
        else:
            T.append(transforms.ConvertVelocitiesFrom(args['units']['dist'], args['units']['time']))

        return dataset_class(**dataset_args, transform=transforms.Compose(T))
            
    def setup_dataset(self):
        dataset_params = self.config_file.dataset.params
            
        if self.config_file.dataset.type == 'compose':
            n_dataset = self.config_file.dataset.n_dataset
            datasets = []
            for i in dataset_args.keys():
                datasets.append(self._setup_dataset(dataset_params[i]))
            from alchemist.data.base import ComposeDatasets
            self.dataset = ComposeDatasets(datasets)
        else:
            self.dataset = self._setup_dataset(dataset_params)
            
        if self.ddp:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
            self.train_loader = DataLoader(self.dataset, batch_size=self.config_file.dataset.batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.sampler, drop_last=False)
        else:
            self.train_loader = DataLoader(self.dataset, batch_size=self.config_file.dataset.batch_size, shuffle=False)
        
    def setup_model(self):
        self.start_epoch = 0
        networks = []
        for i in range(self.n_iter): networks.append(EGCL(node_nf, node_nf, self.hidden_nf))
        integrator_class = getattr(importlib.import_module("alchemist.flow.dynamics"), f"{self.integrator.upper()}Integrator")
        self.model = integrator_class(network=network, n_iter=self.config.flow.network.n_iter, dt=self.config.flow.dt).to(self.local_rank)
    
        if self.config.flow.checkpoint:
            self.checkpoint_path = self.config.flow.checkpoint
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        
        if self.ddp: self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def setup_optim(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.training.lr))
    
        if self.config.training.scheduler_type:
            scheduler_class = getattr(importlib.import_module("torch.optim"), self.config.training.scheduler_type)
            self.scheduler = scheduler_class(self.optimizer, **self.config.training.scheduler_params)
        else:
            scheduler_step = 0
            gamma = 0
        
        if scheduler_step != 0 and gamma != 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
        
        if self.world_rank == 0: eprint(f"Loss function parameters: softening={softening}, kBT={lj_kBT}", flush=True)
        self.nll = Alchemical_NLL(kBT=lj_kBT, softening=softening)
    
        self.nll = Alchemical_NLL(kBT=self.config.loss.temp, softening=self.config.loss.softening)

        if checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def setup(self):
        self.setup_dataset()
        self.setup_model()
    
    def train(self):
        if self.world_rank == 0:
            print('Epoch \tTraining Loss \t   Time (s)', flush=True)

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            losses = []

            if self.ddp: self.sampler.set_epoch(epoch)
            self.model.train()
    
            if self.world_rank == 0:
                eprint(f"###### Starting epoch {epoch} ######", flush=True)
                if self.ddp: torch.cuda.synchronize()
                start_time = time.time()
            
            for i, data in enumerate(self.train_loader):
                if self.world_rank == 0:
                    eprint(f'*** Batch Number {i} out of {len(self.train_loader)} batches ***', flush=True)
                    eprint('GPU \tTraining Loss\t Learning Rate', flush=True)
            
                data = data.to(self.local_rank)
                self.optimizer.zero_grad()
                out, ldj = self.model(data)
                loss = self.nll(out, ldj)
                loss.backward()
                self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                losses.append(loss)
        
                eprint('%.5i \t    %.2f' % (self.world_rank, loss.item()), flush=True)
            
            losses = torch.tensor(losses, device=self.local_rank)
            epoch_loss = torch.mean(losses)
            if self.ddp:
                eprint('Epoch loss on rank %.5i :    %.2f' % (self.world_rank,  epoch_loss.item()), flush=True)
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                epoch_loss /= self.world_size
            
            if self.world_rank == 0:
                to_save = {
                       'epoch': epoch,
                       'model_state_dict': self.model.module.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict()
                   }
                if self.scheduler: to_save['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(to_save, self.checkpoint_path)

                eprint("State saved", flush=True)
        
                eprint(f"###### Ending epoch {epoch} ###### ")
                
                if self.ddp: torch.cuda.synchronize()
                end_time = time.time()
                
                if epoch % self.log_interval == 0: print('%.5i \t    %.2f \t    %.2f \t    %.2e' % (epoch, epoch_loss.item(), end_time - start_time, self.optimizer.param_groups[0]['lr']), flush=True)
    
            if self.ddp: dist.barrier()
    
    def generate(self):
        
        for i, data in enumerate(self.train_loader): 
            if i==0:
                break
        
        out = self.model.reverse(data)

        np.savetxt('h.out', out.h.detach().numpy(), delimiter=' ')

        write_xyz(out, 'test_out.xyz')

        data_, _ = self.model(out)

        print(torch.allclose(data_.pos, data.pos, atol=1e-8))
        print(torch.allclose(data_.h, data.h, atol=1e-8))
       
    def __del__(self):
        if self.ddp: dist.destroy_process_group()
