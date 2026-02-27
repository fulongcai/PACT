import torch

g = torch.Generator()
g.manual_seed(37)

sampler1 = torch.utils.data.RandomSampler(range(10), generator=g)
print(list(sampler1))

torch.use_deterministic_algorithms(True, warn_only=False)

g.manual_seed(37)
sampler2 = torch.utils.data.RandomSampler(range(10), generator=g)
print(list(sampler2))