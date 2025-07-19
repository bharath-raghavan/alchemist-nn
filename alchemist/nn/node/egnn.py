import torch
from torch import nn

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
    
class EGCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Adapted from X
    """

    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.SiLU(), attention=False):
        super().__init__()
        
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
            
    def forward(self, h, edges_index, coord_diff):        
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        
        row = edges_index[0]
        col = edges_index[1]
        
        edge_attr = self.edge_model(h[row], h[col], radial)
        
        return self.node_model(h, row, edge_attr)

class EGNN(nn.Module):
    """Equivariant Graph Neural Net
    Adapted from X
    """

    def __init__(self, node_nf, hidden_nf, n_layers, act_fn=nn.SiLU(), attention=False):
        super().__init__()
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, node_nf)
        networks = []
        for i in range(0, n_layers):
            networks.append(EGCL(hidden_nf, hidden_nf, hidden_nf, act_fn=act_fn, attention=attention))
        self.networks = torch.nn.ModuleList(networks) 
        
    def forward(self, h, edge_index, coord_diff):
        h = self.embedding_in(h)
        for egcl in self.networks:
            h = egcl(h, edge_index, coord_diff)
        h = self.embedding_out(h)
        return h
            
