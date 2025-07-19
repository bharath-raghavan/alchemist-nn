from torch import nn

class ScalarNodeModel(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
    
    def forward(self, h):
        return self.network(h)