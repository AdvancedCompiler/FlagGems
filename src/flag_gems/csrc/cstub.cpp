#include <torch/extension.h>  // 一定要有这个或者 <torch/library.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
namespace py = pybind11;

namespace flag_gems {

// sum算子实现
at::Tensor sum_dim(const at::Tensor& input, c10::OptionalArrayRef<int64_t> dims, bool keepdim, c10::optional<at::ScalarType> dtype) {
    return at::sum(input, dims, keepdim, dtype);
}

// add算子实现
at::Tensor add_tensor(const at::Tensor& a, const at::Tensor& b) {
    return a + b;
}

// rms_norm算子实现，注意epsilon改为double
at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
    // TODO: 完善你的具体实现
    return input;
}

// fused_add_rms_norm算子实现，注意epsilon改为double
void fused_add_rms_norm(at::Tensor& input, at::Tensor& residual, const at::Tensor& weight, double epsilon) {
    input.add_(residual);
}

} // namespace flag_gems

// pybind11模块绑定（可选，方便后续Python调用）
PYBIND11_MODULE(c_operators, m) {
    m.doc() = "flag_gems custom operators";
}

// 注册算子接口和元信息
TORCH_LIBRARY(flag_gems, m) {
    m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    m.def("add_tensor(Tensor a, Tensor b) -> Tensor");
    m.def("rms_norm(Tensor input, Tensor weight, double epsilon) -> Tensor");
    m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, double epsilon) -> ()");
}

// 类型别名，方便转换
using rms_norm_fn = at::Tensor(const at::Tensor&, const at::Tensor&, double);
using fused_add_rms_norm_fn = void(at::Tensor&, at::Tensor&, const at::Tensor&, double);

// CPU设备上的实现绑定
TORCH_LIBRARY_IMPL(flag_gems, CPU, m) {
  m.impl("sum.dim_IntList", TORCH_FN(flag_gems::sum_dim));
  m.impl("add_tensor", TORCH_FN(flag_gems::add_tensor));
  m.impl("rms_norm", TORCH_FN(static_cast<rms_norm_fn*>(&flag_gems::rms_norm)));
  m.impl("fused_add_rms_norm", TORCH_FN(static_cast<fused_add_rms_norm_fn*>(&flag_gems::fused_add_rms_norm)));
}

// 你如果后续有CUDA版本，可以再写TORCH_LIBRARY_IMPL(flag_gems, CUDA, m)区块
