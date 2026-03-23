import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplaneNeRF(nn.Module):
    def __init__(self, plane_resolution: int = 64, plane_channels: int = 32, mlp_hidden: int = 128):
        super().__init__()
        self.plane_resolution = plane_resolution
        self.plane_channels = plane_channels
        self.planes = nn.Parameter(torch.randn(3, plane_channels, plane_resolution, plane_resolution) * 0.01)
        feat_dim = plane_channels * 3
        self.density_mlp = nn.Sequential(nn.Linear(feat_dim, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 1), nn.Softplus())
        self.rgb_mlp = nn.Sequential(nn.Linear(feat_dim, mlp_hidden), nn.ReLU(inplace=True), nn.Linear(mlp_hidden, 3), nn.Sigmoid())

    def sample_plane(self, plane: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
        grid = coords_2d.reshape(1, 1, -1, 2)
        plane_4d = plane.unsqueeze(0)
        sampled = F.grid_sample(plane_4d, grid, mode="bilinear", align_corners=True, padding_mode="border")
        return sampled.squeeze(0).squeeze(1).t()

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = points.shape
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        all_rgb, all_density = [], []
        for b in range(B):
            xy = torch.stack([x[b], y[b]], dim=-1)
            xz = torch.stack([x[b], z[b]], dim=-1)
            yz = torch.stack([y[b], z[b]], dim=-1)
            f_xy = self.sample_plane(self.planes[0], xy)
            f_xz = self.sample_plane(self.planes[1], xz)
            f_yz = self.sample_plane(self.planes[2], yz)
            feat = torch.cat([f_xy, f_xz, f_yz], dim=-1)
            density = self.density_mlp(feat)
            rgb = self.rgb_mlp(feat)
            all_rgb.append(rgb)
            all_density.append(density)
        return torch.stack(all_rgb), torch.stack(all_density)

    def get_triplane_features(self) -> torch.Tensor:
        return self.planes.data.clone()
