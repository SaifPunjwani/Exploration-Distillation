import torch
import torch.nn as nn
import torch.nn.functional as F

from .devices import optimizer_step, mark_step


class RNDModule(nn.Module):
    """
    Random Network Distillation:
      - target: random fixed net
      - predictor: trainable net
      - novelty = distance(target(h), predictor(h)) in feature space.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, lr: float = 1e-4, device=None, is_xla: bool = False):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        for p in self.target.parameters():
            p.requires_grad = False

        if device is not None:
            self.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.is_xla = is_xla

    def compute_novelty(self,
                        x: torch.Tensor,
                        train: bool = True,
                        metric: str = "mse",
                        clip_value=None,
                        return_vec: bool = False) -> torch.Tensor:
        """
        x: [H] or [B, H]
        If train=True: update predictor (used during RL).
        If train=False: just compute distance (used for analysis/eval).
        """
        metric = (metric or "mse").lower()

        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, H]

        if train:
            with torch.no_grad():
                t = self.target(x)
            pred = self.predictor(x)

            diff = pred - t
            mse_vec = F.mse_loss(pred, t, reduction="none").mean(dim=1)

            if metric == "l2":
                novelty_vec = diff.pow(2).sum(dim=1).sqrt()
            elif metric == "sqrt_mse":
                novelty_vec = mse_vec.sqrt()
            elif metric == "l1":
                novelty_vec = diff.abs().mean(dim=1)
            elif metric == "cosine":
                novelty_vec = 1.0 - F.cosine_similarity(pred, t, dim=1)
            else:
                novelty_vec = mse_vec

            if clip_value is not None:
                novelty_vec = novelty_vec.clamp(max=clip_value)

            loss = mse_vec.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_step(self.optimizer, is_xla=self.is_xla)
            mark_step(is_xla=self.is_xla)
            novelty_vec = novelty_vec.detach()
            del loss
            del mse_vec
            del diff
            del pred
            del t
            return novelty_vec if return_vec else novelty_vec.mean()
        else:
            with torch.no_grad():
                t = self.target(x)
                pred = self.predictor(x)
                diff = pred - t
                mse_vec = F.mse_loss(pred, t, reduction="none").mean(dim=1)

                if metric == "l2":
                    novelty_vec = diff.pow(2).sum(dim=1).sqrt()
                elif metric == "sqrt_mse":
                    novelty_vec = mse_vec.sqrt()
                elif metric == "l1":
                    novelty_vec = diff.abs().mean(dim=1)
                elif metric == "cosine":
                    novelty_vec = 1.0 - F.cosine_similarity(pred, t, dim=1)
                else:
                    novelty_vec = mse_vec

                if clip_value is not None:
                    novelty_vec = novelty_vec.clamp(max=clip_value)

            return novelty_vec if return_vec else novelty_vec.mean()
