import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ExpandNet(nn.Module):
    def __init__(self):
        super(ExpandNet, self).__init__()

        def layer(nIn, nOut, k, s, p, d=1):
            return nn.Sequential(
                nn.Conv2d(nIn, nOut, k, s, p, d), nn.SELU(inplace=True)
            )

        self.nf = 64
        self.local_net = nn.Sequential(
            layer(3, 64, 3, 1, 1), layer(64, 128, 3, 1, 1)
        )

        self.mid_net = nn.Sequential(
            layer(3, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            nn.Conv2d(64, 64, 3, 1, 2, 2),
        )

        self.glob_net = nn.Sequential(
            layer(3, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 4, 1, 0),
        )

        self.end_net = nn.Sequential(
            layer(256, 64, 1, 1, 0), nn.Conv2d(64, 3, 1, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        local = self.local_net(x)
        mid = self.mid_net(x)
        resized = F.interpolate(
            x, (256, 256), mode='bilinear', align_corners=False
        )
        b, c, h, w = local.shape
        glob = self.glob_net(resized).expand(b, 64, h, w)
        fuse = torch.cat((local, mid, glob), -3)
        return self.end_net(fuse)

    # This uses stitching is for low memory usage
    def predict(self, x, patch_size):
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.size(-3) == 1:
                # For grey images
                x = x.expand(1, 3, *x.size()[-2:])
            # Evaluate global features
            resized = F.interpolate(
                x, (256, 256), mode='bilinear', align_corners=False
            )
            glob = self.glob_net(resized)

            overlap = 20
            skip = int(overlap / 2)

            result = x.clone()
            x = F.pad(x, (skip, skip, skip, skip))
            padded_height, padded_width = x.size(-2), x.size(-1)
            num_h = int(np.ceil(padded_height / (patch_size - overlap)))
            num_w = int(np.ceil(padded_width / (patch_size - overlap)))
            for h_index in range(num_h):
                for w_index in range(num_w):
                    h_start = h_index * (patch_size - overlap)
                    w_start = w_index * (patch_size - overlap)
                    h_end = min(h_start + patch_size, padded_height)
                    w_end = min(w_start + patch_size, padded_width)
                    x_slice = x[:, :, h_start:h_end, w_start:w_end]
                    loc = self.local_net(x_slice)
                    mid = self.mid_net(x_slice)
                    exp_glob = glob.expand(
                        1, 64, h_end - h_start, w_end - w_start
                    )
                    fuse = torch.cat((loc, mid, exp_glob), 1)
                    res = self.end_net(fuse).data
                    # stitch
                    h_start_stitch = h_index * (patch_size - overlap)
                    w_start_stitch = w_index * (patch_size - overlap)
                    h_end_stitch = min(
                        h_start + patch_size - overlap, padded_height
                    )
                    w_end_stitch = min(
                        w_start + patch_size - overlap, padded_width
                    )
                    res_slice = res[:, :, skip:-skip, skip:-skip]
                    result[
                        :,
                        :,
                        h_start_stitch:h_end_stitch,
                        w_start_stitch:w_end_stitch,
                    ].copy_(res_slice)
                    del fuse, loc, mid, res
            return result[0]
