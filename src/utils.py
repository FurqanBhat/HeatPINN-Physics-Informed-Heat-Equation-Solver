import torch

def gradient(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

def double_derivative(y, x):
    dy_dx = gradient(y,x)
    d2y_dx2 = gradient(dy_dx, x)
    return d2y_dx2