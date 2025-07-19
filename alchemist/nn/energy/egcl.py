import torch
from egnn.models.gcl import E_GCL
    
class EGCL(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Adapted from X
    """

    def __init__(self, input_nf, output_nf, hidden_nf, act_fn='silu', attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=act_fn, recurrent=False, coords_weight=1.0, attention=attention, clamp=False, norm_diff=False, tanh=False)

        del self.coord_mlp        

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        pass
    
    def coord2radial(self, edge_index, coord_diff):
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        return radial
            
    def forward(self, h, edge_index, coord_diff):
        row, col = edge_index
        radial = self.coord2radial(edge_index, coord_diff)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        return h
            
