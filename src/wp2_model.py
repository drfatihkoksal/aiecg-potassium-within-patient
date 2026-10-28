"""1D ResNet reference model (WP2 primary architecture).

Deliberately conventional: this audits a *representative* published model, not a
straw man, so the architecture and recipe follow the standard AI-ECG lab-value
setup (12-lead 10 s 500 Hz in, MSE on the continuous value, auxiliary binary
heads for the dyskalaemia classes).
"""
import torch, torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, k=7, p_drop=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.do = nn.Dropout(p_drop)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(cout))
                     if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.b1(self.c1(x)))
        x = self.do(x)
        x = self.b2(self.c2(x))
        return self.act(x + r)


class ECGResNet(nn.Module):
    """~7 M parameter 1D ResNet, 4 stages x 2 blocks, 64->512 channels."""

    def __init__(self, n_leads=12, widths=(64, 128, 256, 512), n_aux=2, p_drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages, cin = [], widths[0]
        for i, w in enumerate(widths):
            stages += [BasicBlock(cin, w, stride=1 if i == 0 else 2, p_drop=p_drop),
                       BasicBlock(w, w, p_drop=p_drop)]
            cin = w
        self.stages = nn.Sequential(*stages)
        self.head_reg = nn.Sequential(nn.Linear(2 * cin, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(p_drop), nn.Linear(256, 1))
        self.head_aux = nn.Sequential(nn.Linear(2 * cin, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(p_drop), nn.Linear(256, n_aux))

    def embed(self, x):
        h = self.stages(self.stem(x))
        return torch.cat([h.mean(-1), h.amax(-1)], dim=1)

    def forward(self, x):
        z = self.embed(x)
        return self.head_reg(z).squeeze(-1), self.head_aux(z)
