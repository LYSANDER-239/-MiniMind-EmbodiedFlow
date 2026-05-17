import torch
import torch.nn as nn


class ConditionalFlowMLP(nn.Module):
    def __init__(
        self,
        traj_len: int = 32,
        condition_dim: int = 14,
        hidden_dim: int = 512,
        num_layers: int = 5,
    ):
        super().__init__()
        input_dim = traj_len * 2 + 1 + condition_dim
        output_dim = traj_len * 2
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.traj_len = traj_len
        self.condition_dim = condition_dim

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = z_t.shape[0]
        if t.ndim == 1:
            t = t.view(batch_size, 1)
        z_t_flat = z_t.reshape(batch_size, -1)
        x = torch.cat([z_t_flat, t, condition], dim=-1)
        out = self.net(x)
        return out.view(batch_size, self.traj_len, 2)
