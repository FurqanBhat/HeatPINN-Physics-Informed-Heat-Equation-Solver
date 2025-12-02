import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from pinn import PINN
from dataset import sample_points, initial_condition_points, boundary_condition_points
from utils import gradient, double_derivative


def pde_residual(x, t, model):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)
    u_t = gradient(u, t)
    u_xx = double_derivative(u, x)

    return u_t - u_xx


def train():

    # ---- Hyperparameters ----
    num_epochs = 1500
    n_sp = 5000
    n_icp = 500
    n_bcp = 500

    lr = 1e-3
    alpha_sp = 1
    alpha_icp = 1
    alpha_bcp = 1

    # ---- Model ----
    layers = [2, 8, 16, 8, 1]
    model = PINN(layers)

    # ---- Optimizer & loss ----
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    loss_history = []

    # ---- Training loop ----
    model.train()
    for epoch in range(num_epochs):

        # PDE residual collocation points
        x_sp, t_sp = sample_points(n_sp=n_sp)
        residue = pde_residual(x_sp, t_sp, model)
        residual_loss = mse_loss(residue, torch.zeros_like(residue))

        # Initial condition
        x_ic, t_ic, u_ic = initial_condition_points(n_icp=n_icp)
        ic_pred = model(x_ic, t_ic)
        ic_loss = mse_loss(ic_pred, u_ic)

        # Boundary condition
        x0, t_bc, u0, x1, _, u1 = boundary_condition_points(n_bcp=n_bcp)
        u0_pred = model(x0, t_bc)
        u1_pred = model(x1, t_bc)
        bc_loss = mse_loss(u0_pred, u0) + mse_loss(u1_pred, u1)

        # Total loss
        total_loss = alpha_sp*residual_loss + alpha_icp*ic_loss + alpha_bcp*bc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}/{num_epochs} "
                f"Loss: {total_loss.item():.4e}  "
                f"(PDE: {residual_loss.item():.4e}, "
                f"IC: {ic_loss.item():.4e}, "
                f"BC: {bc_loss.item():.4e})"
            )

    # ---- Save model ----
    torch.save(model.state_dict(), "outputs/saved_models/model.pth")
    print("Model saved to outputs/saved_models/model.pth")
    # ---- Save training loss plot ----
    save_training_loss_plot(loss_history)

    
def save_training_loss_plot(loss_history):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/plots/training_loss.png", dpi=300)
    plt.close()

    print("Training loss curve saved to outputs/plots/training_loss.png")


