import pyro.distributions as dist
import pyro
import torch


class Discrete_Uniform(dist.TorchDistribution):
    arg_constraints = {}

    def __init__(self, vals):
        self.vals = vals
        probs = torch.ones(len(vals))
        self.categorical = dist.Categorical(probs)
        super(Discrete_Uniform, self).__init__(self.categorical.batch_shape,
                                               self.categorical.event_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.vals[self.categorical.sample(sample_shape)]

    def log_prob(self, value):
        idx = (self.vals == value).nonzero()
        return self.categorical.log_prob(idx)


values = [1, 2, 3, 4]
frequency = {el: 0 for el in values}
for i in range(1000):
    sample = pyro.sample('sample', Discrete_Uniform(values))
    if sample == 1:
        frequency[1] += 1
    elif sample == 2:
        frequency[2] += 1
    if sample == 3:
        frequency[3] += 1
    if sample == 4:
        frequency[4] += 1


print(frequency)