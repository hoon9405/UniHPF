import torch
import torch.nn.functional as F

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

def sample_word(logits, strategy='topp', **kwargs):
    if strategy == 'topp':
        return sample_top_p(logits, **kwargs)
    else:
        raise NotImplementedError

def sample_top_p(logits, k=5, sample=False, **kwargs):
    if sample:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("Inf")    
        probs = F.softmax(out, dim=-1)

        ix = torch.multinomial(probs.squeeze(0), num_samples=1)
    else:
        _, ix = torch.topk(logits, k=1, dim=-1)
    
    next_words = ix

    return next_words

def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}
    
    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    
    return _apply(sample)

def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()
    return apply_to_sample(_move_to_cpu, sample)