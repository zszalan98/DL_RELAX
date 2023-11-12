# Create mask generator
# This function implements the generator for creating masks in the RELAX Framework
# The RELAX framework implementation can be found at https://github.com/Wickstrom/RELAX.
# The implementation is based on the TorchRay implementation and the
# RISE implementation at: https://github.com/eclique/RISE.

def MaskGenerator(iterations, inp_shape, num_cells=7, mask_bs=2, p=0.5, nsd=2, dev='cpu'):
    for iteration in range(iterations):

        pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)
        grid = (torch.rand(mask_bs, 1, *((num_cells,) * nsd), device=dev) < p).float()

        grid_up = F.interpolate(grid, size=(inp_shape), mode='bilinear', align_corners=False)
        grid_up = F.pad(grid_up, pad_size, mode='reflect')

        shift_x = torch.randint(0, num_cells, (mask_bs,), device=dev)
        shift_y = torch.randint(0, num_cells, (mask_bs,), device=device)

        masks = torch.empty((mask_bs, 1, inp_shape, inp_shape), device=dev)

        for i in range(mask_bs):
            masks[i] = grid_up[i, :,
                               shift_x[i]:shift_x[i] + inp_shape,
                               shift_y[i]:shift_y[i] + inp_shape]

        yield masks

