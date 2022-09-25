import torch
import cppcuda_tutorial
import time


def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]
    
    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp


if __name__ == '__main__':
    N = 65536; F = 256
    feats = torch.rand(N, 8, F, device='cuda').requires_grad_()
    points = torch.rand(N, 3, device='cuda')*2-1

    t = time.time()
    out_cuda = cppcuda_tutorial.trilinear_interpolation(feats, points)
    torch.cuda.synchronize()
    print('   cuda time', time.time()-t, 's')

    t = time.time()
    out_py = trilinear_interpolation_py(feats, points)
    torch.cuda.synchronize()
    print('pytorch time', time.time()-t, 's')

    print(torch.allclose(out_py, out_cuda))