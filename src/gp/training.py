import botorch
import gpytorch
import numpy as np
import torch
import tqdm
from botorch.fit import fit_gpytorch_mll

botorch.settings.debug = True


def train_exact_gp_adam(
    model: gpytorch.models.ExactGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    max_iter: int = 100,
    rel_ftol: float = 0.0,
    loss_scale: int = 1,
    lr: float = 1,
    messages: bool = True,
):
    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    output = model(train_x)
    loss = -mll(output, train_y)

    if messages:
        print(f"start loss={loss.detach().cpu().item() * loss_scale}")

    losses = np.zeros((max_iter + 1)) * np.nan

    for i in (pbar := tqdm.trange(max_iter, disable=not messages)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        current_loss = loss.detach().cpu().item() * loss_scale

        losses[i] = current_loss

        if i > 0:
            pbar.set_description(f"loss={current_loss}")

            prev_loss = losses[i - 1]
            if abs((current_loss - prev_loss) / prev_loss) < rel_ftol:
                losses = losses[: i + 1]
                break

        optimizer.step()

    loss = -mll(output, train_y)
    final_loss = loss.detach().cpu().item() * loss_scale
    losses[-1] = final_loss

    if messages:
        print(f"final loss={final_loss}")

    model.eval()
    model.likelihood.eval()

    return losses


def train_exact_gp_botorch(
    model: gpytorch.models.ExactGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    **_kwargs,
):
    model.train()
    model.likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # Set dtype and device
    mll = mll.to(train_x)
    fit_gpytorch_mll(
        mll,
        max_retries=10000,
        optimizer_kwargs={
            "options": {
                "maxiter": 10000,
                "ftol": 1e-15,
                "gtol": 1e-15,
                "maxfun": 10000,
                "maxls": 10000,
                "eps": 1e-15,
            }
        },
    )

    model.eval()
    model.likelihood.eval()
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    output = model(train_x)
    loss = -mll(output, train_y)
    print(f"start loss={loss.detach().cpu().item()}")
    return loss.detach().cpu().item()


def train_exact_gp_lbfgs(
    model: gpytorch.models.ExactGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    max_iter: int = 20,
    rel_ftol: float = 1e-4,
    loss_scale: int = 1,
    lr: float = 1,
    messages: bool = True,
):
    model.train()
    model.likelihood.train()

    optimizer = torch.optim.LBFGS(
        model.parameters(), line_search_fn="strong_wolfe", lr=lr
    )

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    output = model(train_x)
    loss = -mll(output, train_y)

    if messages:
        print(f"start loss={loss.detach().cpu().item() * loss_scale}")

    losses = np.zeros((max_iter + 1)) * np.nan

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss

    for i in (pbar := tqdm.trange(max_iter, disable=not messages)):
        output = model(train_x)
        loss = -mll(output, train_y)

        current_loss = loss.detach().cpu().item() * loss_scale

        losses[i] = current_loss

        if i > 0:
            pbar.set_description(f"loss={current_loss}")

            prev_loss = losses[i - 1]
            if abs((current_loss - prev_loss) / prev_loss) < rel_ftol:
                losses = losses[: i + 1]
                break

        optimizer.step(closure)

    loss = -mll(output, train_y)
    final_loss = loss.detach().cpu().item() * loss_scale
    losses[-1] = final_loss

    if messages:
        print(f"final loss={final_loss}")

    model.eval()
    model.likelihood.eval()

    return losses
