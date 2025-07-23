import torch

class Default(torch.nn.Module):
    def __init__(self, node_nf, dtype, dim):
        super().__init__()
        self.network = torch.nn.Embedding(node_nf, dim, dtype=dtype)
        self.dim = dim
        
    def forward(self, z):
        return self.network(z), 0
    
    def reverse(self, x):
        emb_weights = self.network.weight
        output = x.unsqueeze(0)
        
        emb_size = output.size(0), output.size(1), -1, -1
        out_size = -1, -1, emb_weights.size(0), -1
        z = torch.argmin(torch.abs(output.unsqueeze(2).expand(out_size) -
                                        emb_weights.unsqueeze(0).unsqueeze(0).expand(emb_size)).sum(dim=3), dim=2)[0]
        return z