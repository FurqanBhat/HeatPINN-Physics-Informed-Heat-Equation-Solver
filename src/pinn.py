import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        net=[]
        for i in range(len(layers)-1):
            input=layers[i]
            linear=nn.Linear()
            self.net.append(linear)
            if i<(len(layers)-2):
                self.net.append(nn.Tanh())
            
        #[2, 8, 16, 8, 1]
        self.net = nn.Sequential(*net)


    def forward(self, x, t):
        """
        x: (N, 1)
        t: (N, 1)
        inp: (N, 2)
        returns u: (N, 1)
        """

        inp = torch.cat([x, t], dim=1)
        return self.net(inp)



