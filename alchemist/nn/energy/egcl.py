import torch
from torch import nn
from torch.autograd import grad
from ...utils.helpers import unsorted_segment_sum, unsorted_segment_mean
    
class EGCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Adapted from X
    """


    def __init__(self, h_dim, box, hidden_nf, act_fn='silu', coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False, max_num_neighbors=32, add_self_loops=False):
        super().__init__()
        input_nf = h_dim
        output_nf = h_dim
        
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        
        if act_fn == 'silu': act_fn = nn.SiLU()
        
        self.edge_nn = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_nn = []
        coord_nn.append(nn.Linear(hidden_nf, hidden_nf))
        coord_nn.append(act_fn)
        coord_nn.append(layer)
        if self.tanh:
            coord_nn.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_nn = nn.Sequential(*coord_nn)


        if self.attention:
            self.att_nn = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            
        self.norm_diff = norm_diff
        
        self.distance = PeriodicDistance(cutoff, box, max_num_neighbors=max_num_neighbors, add_self_loops=add_self_loops)

    def edge_model(self, source, target, radial):
        out = torch.cat([source, target, radial], dim=1)
        out = self.edge_nn(out)
        if self.attention:
            att_val = self.att_nn(out)
            out = out * att_val
        return out

    def energy_model(self, coord_diff, row, edge_feat, num_segments):
        trans = coord_diff * self.coord_nn(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it explodes it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=num_segments)
        return agg*self.coords_weight
    
    def forward(self, data):
        edge_index, coord_diff = self.distance(data.pos, data.batch)
        
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)
        
        row = edge_index[0]
        col = edge_index[1]
            
        edge_attr = self.edge_model(data.h[row], data.h[col], radial)

        return self.energy_model(coord_diff, row, edge_attr, data.h.size(0))
            
