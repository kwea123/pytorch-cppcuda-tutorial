#include <torch/extension.h>


torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
    
}