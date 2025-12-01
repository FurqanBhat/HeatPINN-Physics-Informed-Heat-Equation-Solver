import torch


def sample_points(n_sp):
    x_sp=torch.rand(n_sp, 1)
    t_sp=torch.rand(n_sp, 1)
    return x_sp, t_sp


def initial_condition_points(n_icp):
    x_icp=torch.rand(n_icp,1)
    t_icp=torch.zeros_like(x_icp)
    u_icp=torch.sin(torch.pi*x_icp)

    return x_icp, t_icp, u_icp

def boundary_condition_points(n_bcp):
    t_bcp=torch.rand(n_bcp,1)
    x0_bcp=torch.zeros_like(t_bcp)
    x1_bcp=torch.ones_like(t_bcp)
    u0_bcp=torch.zeros_like(t_bcp)
    u1_bcp=torch.zeros_like(t_bcp)

    return (
        x0_bcp, t_bcp, u0_bcp,
        x1_bcp, t_bcp, u1_bcp
    )
    