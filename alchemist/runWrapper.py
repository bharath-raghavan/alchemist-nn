import os
import sys
import numpy as np
from datetime import timedelta
import time
import importlib

import torch

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

def write_xyz(out, elems, file):
    with open(file, 'w') as f:
        f.write("%d\n%s\n" % (out.N.item(), ' '))
        for x, a in zip(out.pos, elems):
            x = x*sigma*1e10
            f.write("%s %.18g %.18g %.18g\n" % (a, x[0].item(), x[1].item(), x[2].item()))
                
class RunWrapper:

    def __init__(self, world_size=None, world_rank=None, local_rank=None, num_cpus_per_task=None):
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
        
    def setup(self, config, train=True):
        if train:
            self.dataset = config.dataset.get()
        else:
            self.dataset = config.generate.get()
            
        if self.ddp:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
            self.train_loader = DataLoader(self.dataset, batch_size=config.dataset.batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.sampler, drop_last=False)
        else:
            self.train_loader = DataLoader(self.dataset, batch_size=config.dataset.batch_size, shuffle=False)

        self.model = config.flow.get(self.dataset.node_nf).to(self.local_rank)
        
        self.checkpoint_path = config.checkpoint
            
        if os.path.exists(self.checkpoint_path) and self.checkpoint_path != None:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        else:
            checkpoint = None
        
        if self.ddp: self.model = DDP(self.model, device_ids=[self.local_rank])
    
        if train:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.training.lr))
    
            if config.training.scheduler_type:
                scheduler_class = getattr(importlib.import_module("torch.optim.lr_scheduler"), config.training.scheduler_type)
                self.scheduler = scheduler_class(self.optimizer, **config.training.scheduler_params)
            else:
                self.scheduler = None
                scheduler_step = 0
                gamma = 0

            self.nll = config.training.loss.get()

            if checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.start_epoch = 0
            self.num_epochs = self.start_epoch+config.training.num_epochs
            self.log_interval = config.training.log_interval
        else:
            self.atom_types_list = config.generate.params.atom_types
    
    def _train_backprop(self, data, accum_iter, batch_idx):
        # passes and weights update
        with torch.set_grad_enabled(True):
            out, ldj = self.model(data)
            loss = self.nll(out, ldj)
            loss = loss/accum_iter
            loss.backward()

            # weights update
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(self.train_loader)):
                self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                self.optimizer.zero_grad()

        return loss.item()

    def train(self):
        if self.world_rank == 0:
            print('Epoch \tTraining Loss \t   Time (s)', flush=True)

        for epoch in range(self.start_epoch, self.num_epochs):
            losses = 0

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
                loss = self._train_backprop(data, 10, i)
                losses += loss

                eprint('%.5i \t    %.2f' % (self.world_rank, loss), flush=True)

            epoch_loss = torch.tensor(losses/len(self.train_loader), device=self.local_rank)
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
                
                if self.checkpoint_path != None:
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

        atom_types = {i:z for i,z in enumerate(self.atom_types_list)}
        elems = [atom_types[i.item()] for i in out.h.argmax(dim=1)]
        
        write_xyz(out, elems, 'test_out.xyz')

        data_, _ = self.model(out)
        print(data)
        print(torch.allclose(data_.pos, data.pos, atol=1e-8))
        print(torch.allclose(data_.h, data.h, atol=1e-8))
       
    def __del__(self):
        if self.ddp: dist.destroy_process_group()
