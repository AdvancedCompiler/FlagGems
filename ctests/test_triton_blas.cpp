#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(blas_op_test, mm) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);

  torch::Tensor out_torch = at::mm(a, b);
  torch::Tensor out_triton = flag_gems::mm_tensor(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}

TEST(blas_op_test, true_div) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);  // avoid divide by zero

  auto out_torch = a / b;
  auto out_triton = flag_gems::true_div(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, trunc_div) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = torch::trunc(a / b);
  auto out_triton = flag_gems::trunc_div(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, floor_div) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = torch::floor_divide(a, b);
  auto out_triton = flag_gems::floor_div(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, div_mode) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = at::div(a, b);
  auto out_triton = flag_gems::div_mode(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, remainder) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({32, 32}, device) * 10;
  torch::Tensor b = torch::randn({32, 32}, device).clamp_min(0.5);

  auto out_torch = torch::remainder(a, b);
  auto out_triton = flag_gems::remainder(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}
