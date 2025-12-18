import torch

from precision_track.registry import MODELS
from precision_track.utils import get_device


@MODELS.register_module()
class BaseVelocityEncoder:
    def __init__(self):
        super().__init__()

    def __call__(self, velocities):
        return velocities

    @property
    def nb_dim(self):
        return 2


@MODELS.register_module()
class VelocityRBFEncoder(BaseVelocityEncoder):
    def __init__(self, config):
        super().__init__()
        self.num_rbf = int(config.num_rbf)
        self.eps = 1e-6
        self.max_rel_vel = config.max_rel_vel

        self.device = config.get("device", None) or get_device()

        self.mu = torch.linspace(0, self.max_rel_vel, self.num_rbf, device=self.device)
        self.sigma = torch.full((self.num_rbf,), 0.5, device=self.device)

    def __call__(self, velocities):
        if velocities.ndim == 2:
            velocities = velocities.unsqueeze(0)
        velocities = velocities.norm(p=2, dim=-1).clamp(max=self.max_rel_vel)
        return torch.exp(-0.5 * ((velocities.unsqueeze(-1) - self.mu) / self.sigma) ** 2)

    @property
    def nb_dim(self):
        return self.num_rbf


@MODELS.register_module()
class VelocityAbsEncoder(BaseVelocityEncoder):
    def __init__(self):
        super().__init__()

    def __call__(self, velocities):
        return torch.abs(velocities)

    @property
    def nb_dim(self):
        return 2


@MODELS.register_module()
class VelocityNormEncoder(BaseVelocityEncoder):
    def __init__(self):
        super().__init__()

    def __call__(self, velocities):
        return torch.norm(velocities, p=2, dim=-1)

    @property
    def nb_dim(self):
        return 1
