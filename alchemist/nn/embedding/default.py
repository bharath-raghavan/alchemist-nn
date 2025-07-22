import torch

class Default(torch.nn.Module):
    def __init__(self, node_nf, dtype, hidden_channels):
        super().__init__()
        self.network = torch.nn.Embedding(node_nf, hidden_channels, dtype=dtype)
        
    def forward(self, z):
        return self.network(z), 0
    
    def get_embedding_index(self, x):
        results = torch.where(torch.sum((self.network.weight==x), axis=1))
        if len(results[0])==len(x):
            return None
        else:
            return results[0][0]
    
    def reverse(self, h):
        return torch.Tensor(list(map(self.get_embedding_index, h)))