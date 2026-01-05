import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, ModuleList, BatchNorm1d
from torch_geometric.nn import PointNetConv, fps, radius, knn_interpolate

class SAModuleMSG(torch.nn.Module):
    def __init__(self, ratio, r_list, nn_list):
        super(SAModuleMSG, self).__init__()
        self.ratio = ratio
        self.r_list = r_list
        self.convs = ModuleList()
        
        for nn in nn_list:
            self.convs.append(PointNetConv(nn, add_self_loops=False))

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        new_x_list = []
        for i, r in enumerate(self.r_list):
            row, col = radius(pos, pos[idx], r, batch, batch[idx], max_num_neighbors=128)
            edge_index = torch.stack([col, row], dim=0)
            x_scale = self.convs[i]((x, x[idx]), (pos, pos[idx]), edge_index)
            new_x_list.append(x_scale)
        x = torch.cat(new_x_list, dim=1)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class ForestPointNetPP(torch.nn.Module):
    def __init__(self, num_classes=2, in_channels=5, dropout=0.5): # Added dropout arg
        super(ForestPointNetPP, self).__init__()

        def make_mlp(in_feat, out_feats):
            layers = []
            last = in_feat
            for out in out_feats:
                layers.append(Linear(last, out))
                layers.append(ReLU())
                layers.append(BatchNorm1d(out))
                last = out
            return Sequential(*layers)

        # --- ENCODER ---
        # SA1: 4096 -> 1024 points
        self.sa1 = SAModuleMSG(
            ratio=0.25, 
            r_list=[0.05, 0.4],     
            nn_list=[
                make_mlp(in_channels + 3, [32, 32, 32]),
                make_mlp(in_channels + 3, [32, 32, 32])
            ]
        )

        # SA2: 1024 -> 256 points
        self.sa2 = SAModuleMSG(
            ratio=0.25, 
            r_list=[0.2, 0.8],      
            nn_list=[
                make_mlp(64 + 3, [64, 64, 64]),
                make_mlp(64 + 3, [64, 64, 64])
            ]
        )

        # SA3: 256 -> 64 points
        self.sa3 = SAModuleMSG(
            ratio=0.25, 
            r_list=[0.4, 1.6],      
            nn_list=[
                make_mlp(128 + 3, [128, 128, 128]),
                make_mlp(128 + 3, [128, 128, 128])
            ]
        )

        # --- DECODER ---
        self.fp3 = FPModule(k=3, nn=make_mlp(384, [256, 256]))
        self.fp2 = FPModule(k=3, nn=make_mlp(320, [256, 128]))
        self.fp1 = FPModule(k=3, nn=make_mlp(128 + in_channels, [128, 128, 128]))

        # --- CLASSIFICATION HEAD ---
        # Using the tunable dropout parameter
        self.classifier = Sequential(
             Linear(128, 128), 
             ReLU(), 
             Dropout(dropout), 
             Linear(128, 64), 
             ReLU(), 
             Dropout(dropout), 
             Linear(64, num_classes)
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        sa3_out = self.sa3(*sa2_out)

        fp3_out = self.fp3(sa3_out[0], sa3_out[1], sa3_out[2], sa2_out[0], sa2_out[1], sa2_out[2])
        fp2_out = self.fp2(fp3_out[0], fp3_out[1], fp3_out[2], sa1_out[0], sa1_out[1], sa1_out[2])
        fp1_out = self.fp1(fp2_out[0], fp2_out[1], fp2_out[2], sa0_out[0], sa0_out[1], sa0_out[2])

        x, _, _ = fp1_out
        out = self.classifier(x)
        return F.log_softmax(out, dim=-1)