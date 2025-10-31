import torch
import torch.nn.functional as F


class FGSM:
    def __init__(
        self,
        model,
        eps,
        alpha=None,
        restarts=1,
        mu=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=None,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha if alpha is not None else eps  # FGSM 默认 alpha = eps
        self.restarts = restarts
        self.device = device if device is not None else next(model.parameters()).device

        mu = torch.tensor(mu).view(3, 1, 1).to(self.device)
        std = torch.tensor(std).view(3, 1, 1).to(self.device)
        self.std = std
        self.upper_limit = (1 - mu) / std
        self.lower_limit = (0 - mu) / std

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def __call__(self, X, y):
        model = self.model
        epsilon = self.eps / self.std if isinstance(self.eps, float) else self.eps
        alpha = self.alpha / self.std if isinstance(self.alpha, float) else self.alpha

        max_loss = torch.zeros(y.shape[0]).to(self.device)
        max_delta = torch.zeros_like(X).to(self.device)

        for _ in range(self.restarts):
            delta = torch.zeros_like(X).to(self.device)
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(
                    -epsilon[i][0][0].item(), epsilon[i][0][0].item()
                )
            delta.data = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True

            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                continue

            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0]]
            g = grad[index[0]]
            d = self.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = self.clamp(
                d, self.lower_limit - X[index[0]], self.upper_limit - X[index[0]]
            )
            delta.data[index[0]] = d
            delta.grad.zero_()

            all_loss = F.cross_entropy(model(X + delta), y, reduction="none").detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return X + max_delta
