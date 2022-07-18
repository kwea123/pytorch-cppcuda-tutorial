#include "utils.h"


torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor points
){
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_fw_cu(feats, points);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}
