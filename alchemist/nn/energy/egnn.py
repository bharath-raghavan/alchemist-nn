import torch
from torch_scatter import scatter
from ..node.egcl import EGCL
        
class Egnn(torch.nn.Module):
    """Equivariant Graph Neural Net
    Adapted from X
    """

    def __init__(self, node_nf, hidden_nf=128, n_layers=4, act_fn=torch.nn.SiLU(), attention=False, reduce_op="sum"):
        super().__init__()
        self.embedding_in = torch.nn.Linear(node_nf, hidden_nf)
        self.embedding_out = torch.nn.Linear(hidden_nf, 1)
        self.reduce_op = reduce_op
        networks = []
        for i in range(0, n_layers):
            networks.append(EGCL(hidden_nf, hidden_nf, act_fn=act_fn, attention=attention))
        self.networks = torch.nn.ModuleList(networks)

    def forward(self, data, h, edge_index):
        h = self.embedding_in(h)
        for i, egcl in enumerate(self.networks):
            h = egcl(data.pos, h, edge_index)
        E_atom = self.embedding_out(h)
        E_mol = scatter(E_atom, data.batch, dim=0, reduce=self.reduce_op)
        return E_mol
