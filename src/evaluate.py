# evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pinn import PINN


def exact_solution(x, t):
    return np.exp(-(math.pi**2) * t) * np.sin(math.pi * x)


def evaluate():

    # ---- 1. Load model ----
    layers = [2, 8, 16, 8, 1]
    model = PINN(layers)

    checkpoint = "outputs/saved_models/model.pth"
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded and set to eval mode.")

    # ---- 2. Test grid ----
    nx, nt = 100, 100
    x_vals = np.linspace(0, 1, nx)
    t_vals = np.linspace(0, 1, nt)

    X, T = np.meshgrid(x_vals, t_vals)

    x_grid = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
    t_grid = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)

    # ---- 3. Prediction ----
    with torch.no_grad():
        u_pred = model(x_grid, t_grid).cpu().numpy().reshape(nt, nx)

    # ---- 4. Exact ----
    u_exact = exact_solution(X, T)
    error = np.abs(u_pred - u_exact)

    print("Mean absolute error:", error.mean())

    # ---- 5. Heatmaps ----
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("PINN Prediction")
    plt.imshow(u_pred, extent=[0, 1, 1, 0], aspect="auto", cmap="inferno")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Exact Solution")
    plt.imshow(u_exact, extent=[0, 1, 1, 0], aspect="auto", cmap="inferno")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Absolute Error")
    plt.imshow(error, extent=[0, 1, 1, 0], aspect="auto", cmap="viridis")
    plt.colorbar()

    plt.tight_layout()
    save_plots()
    plt.close()



 
# save figure in plots/

def save_plots():
    import os
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/evaluation_heatmaps.png", dpi=300)



