import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, need_grid=True):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
            :param need_grid: to determine whether the transformer create the sampling grid
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        if need_grid:
            vectors = [torch.arange(0, s) for s in size]
            grids = torch.meshgrid(vectors)
            grid = torch.stack(grids)[[1, 0] if len(size) == 2 else
                                      [1, 0, 2]]  # y, x, z ==> x, y, z
            grid = torch.unsqueeze(grid, 0)  # add batch
            grid = grid.type(torch.FloatTensor)
            self.register_buffer('grid', grid)

        self.need_grid = need_grid

    def forward(self, src, flow, mode='bilinear', align_corners=True):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if self.need_grid:
            new_locs = self.grid + flow
        else:
            new_locs = flow * 1.0

        shape = flow.shape[2:]
        if len(shape) == 2:
            shape = [shape[1], shape[0]]
        elif len(shape) == 3:
            shape = [shape[1], shape[0], shape[2]]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i,
                     ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        return nnf.grid_sample(src,
                               new_locs,
                               mode=mode,
                               align_corners=align_corners)
# import torch
# import torch.nn as nn
# import torch.nn.functional as nnf
#
#
# class SpatialTransformer(nn.Module):
#     """
#     [SpatialTransformer] represesents a spatial transformation block
#     that uses the output from the UNet to preform an grid_sample
#     https://pytorch.org/docs/stable/nn.functional.html#grid-sample
#     """
#     def __init__(self, size, need_grid=True):
#         """
#         Instiatiate the block
#             :param size: size of input to the spatial transformer block
#             :param mode: method of interpolation for grid_sampler
#             :param need_grid: to determine whether the transformer create the sampling grid
#         """
#         super(SpatialTransformer, self).__init__()
#
#         # Create sampling grid
#         if need_grid:
#             vectors = [torch.arange(0, s) for s in size]
#             grids = torch.meshgrid(vectors)
#             grid = torch.stack(grids)[[1, 0] if len(size) == 2 else
#                                       [1, 0, 2]]  # y, x, z ==> x, y, z
#             grid = torch.unsqueeze(grid, 0)  # add batch
#             grid = grid.type(torch.FloatTensor)
#             self.register_buffer('grid', grid)
#
#         self.need_grid = need_grid
#
#     def forward(self, src, flow, mode='bilinear', align_corners=True):
#         """
#         Push the src and flow through the spatial transform block
#             :param src: the original moving image
#             :param flow: the output from the U-Net
#         """
#         if self.need_grid:
#             new_locs = self.grid + flow
#         else:
#             new_locs = flow * 1.0
#
#         shape = flow.shape[2:]
#         if len(shape) == 2:
#             shape = [shape[1], shape[0]]
#         elif len(shape) == 3:
#             shape = [shape[1], shape[0], shape[2]]
#
#         # Need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i,
#                      ...] = 2 * (new_locs[:, i, ...] / (max(shape[i] - 1, 1)) - 0.5)
#
#         if shape[0] == 1:
#             new_locs[:, 2, ...] = 0.0
#         # print(new_locs)
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#
#         return nnf.grid_sample(src,
#                                new_locs,
#                                mode=mode,
#                                align_corners=align_corners)
# class SpatialTransformer(nn.Module):
#     """
#     [SpatialTransformer] represesents a spatial transformation block
#     that uses the output from the UNet to preform an grid_sample
#     https://pytorch.org/docs/stable/nn.functional.html#grid-sample
#     """
#     def __init__(self, size, need_grid=True):
#         """
#         Instiatiate the block
#             :param size: size of input to the spatial transformer block
#             :param mode: method of interpolation for grid_sampler
#             :param need_grid: to determine whether the transformer create the sampling grid
#         """
#         super(SpatialTransformer, self).__init__()
#
#         # Create sampling grid
#         if need_grid:
#             vectors = [torch.arange(0, s) for s in size]
#             grids = torch.meshgrid(vectors)
#             grid = torch.stack(grids)
#             grid = torch.unsqueeze(grid, 0)  # add batch
#             grid = grid.type(torch.FloatTensor)
#             self.register_buffer('grid', grid)
#
#         self.need_grid = need_grid
#
#     def forward(self, src, flow, mode='bilinear', align_corners=True):
#         """
#         Push the src and flow through the spatial transform block
#             :param src: the original moving image
#             :param flow: the output from the U-Net
#         """
#         if self.need_grid:
#             new_locs = self.grid + flow
#         else:
#             new_locs = flow * 1.0
#
#         shape = flow.shape[2:]
#         if len(shape) == 2:
#             shape = [shape[1], shape[0]]
#         elif len(shape) == 3:
#             shape = [shape[1], shape[0], shape[2]]
#
#         # Need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i,
#                      ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
#
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]
#
#         return nnf.grid_sample(src,
#                                new_locs,
#                                mode=mode,
#                                align_corners=align_corners)

# class SpatialTransformer(nn.Module):
#     """
#     N-D Spatial Transformer
#     """
#
#     def __init__(self, size, need_grid=True):
#         super().__init__()
#
#         # self.mode = mode
#
#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors)
#         grid = torch.stack(grids)# [[1, 0]]
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.type(torch.FloatTensor)
#
#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer('grid', grid)
#
#     def forward(self, src, flow, mode='bilinear', align_corners=True):
#         # new locations
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]
#
#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
#
#         # move channels dim to last position
#         # also not sure why, but the channels need to be reversed
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             # new_locs = new_locs[..., [2, 1, 0]]
#
#         return nnf.grid_sample(src,
#                                        new_locs,
#                                        mode=mode,
#                                        align_corners=align_corners)
