import math
import torch

def log_gaussian(z):
    return -0.5*( (z**2).sum() + math.log(2*math.pi) )
    
def apply_pbc(pos, box):
    return pos - (pos/box).round()*box

def get_box_len(pos):
    min_ = torch.min(pos, dim=0)[0]
    max_ = torch.max(pos, dim=0)[0]
    return (max_-min_).round()

def get_element(elem, mass):
    if elem == '':
        mass_int = int(round(mass))
        if mass_int == 1:  # Guess H from mass
            return 'H'
        elif mass_int < 36 and mass_int > 1:  # Guess He to Cl from mass
            return ELEMENTS[mass_int//2]
        else:
            print("error")
    
    return elem
        
def one_hot(index, num_classes=None, dtype=None):
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)
    
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
