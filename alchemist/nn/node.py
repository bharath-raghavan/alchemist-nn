import torch
from torch import nn
from ..utils.helpers import unsorted_segment_sum, unsorted_segment_mean

class NodeModel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Adapted from X
    """

    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
    
    def forward(self, h):
        return self.network(h)

class NodeEdgeModel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Adapted from X
    """

    def __init__(self, h_dim, hidden_nf, act_fn=nn.SiLU(), attention=False):
        super().__init__()
        input_nf = h_dim
        output_nf = h_dim
        
        input_edge = input_nf * 2
        self.attention = attention
        edge_coords_nf = 1

        self.edge_nn = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.node_nn = nn.Sequential(
                    nn.Linear(hidden_nf + input_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, output_nf))
        
        if self.attention:
            self.att_nn = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial):
        out = torch.cat([source, target, radial], dim=1)
        out = self.edge_nn(out)
        if self.attention:
            att_val = self.att_nn(out)
            out = out * att_val
        return out

    def node_model(self, x, row, edge_attr):
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_nn(agg)
        return out
            
    def forward(self, h, edges):
        coord_diff = edges.coord_diff
        
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        
        row = edges.row
        col = edges.col
        
        edge_attr = self.edge_model(h[row], h[col], radial)
        
        return self.node_model(h, row, edge_attr)
            
