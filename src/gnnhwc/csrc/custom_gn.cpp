#include <ATen/core/op_registration/adaption.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include "gn_kernel.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

std::tuple<at::Tensor, at::Tensor, at::Tensor> gn_nhwc_fwd(
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const int64_t G,
    double eps,
    const int64_t act_fn_option) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  const int N = X.size(0);
  const int C = X.size(1);
  const int R = X.size(2);

  // see: https://github.com/pytorch/pytorch/blob/d072254eaea325a507c1498431e4c8294205fe2d/torchgen/dest/register_dispatch_key.py#L275
  std::optional<c10::Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(common_device, X, "gn_nhwc_fwd", "X");
  c10::impl::check_and_update_common_device(common_device, weight, "gn_nhwc_fwd", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "gn_nhwc_fwd", "bias");

  // see: https://github.com/pytorch/pytorch/blob/d072254eaea325a507c1498431e4c8294205fe2d/torchgen/dest/register_dispatch_key.py#L505
  c10::OptionalDeviceGuard guard(at::device_of(X));

  at::Tensor X_nhwc = X.permute({0, 2, 1});
  at::Tensor X_out = at::empty_like(X_nhwc);
  at::Tensor means = at::empty({N, G}, weight.options());
  at::Tensor rstds = at::empty({N, G}, weight.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_forward", [&]() {
    run_gn_fwd_kernels<scalar_t>(
        X_nhwc.const_data_ptr<scalar_t>(),
        weight.const_data_ptr<scalar_t>(), bias.const_data_ptr<scalar_t>(),
        N, R, C, G, static_cast<scalar_t>(eps), act_fn_option,
        X_out.mutable_data_ptr<scalar_t>(), means.mutable_data_ptr<scalar_t>(), rstds.mutable_data_ptr<scalar_t>()
    );
  });
  return {X_out.permute({0, 2, 1}), means, rstds};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gn_nhwc_bwd(
    const at::Tensor dy,
    const at::Tensor X,
    const at::Tensor weight,
    const at::Tensor bias,
    const at::Tensor means,
    const at::Tensor rstds,
    const int64_t G,
    const int64_t act_fn_option) {
  CHECK_CUDA(dy);
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  CHECK_CUDA(means);
  CHECK_CUDA(rstds);

  const int N = X.size(0);
  const int C = X.size(1);
  //const int H = X.size(2);
  //const int W = X.size(3);
  const int R = X.size(2);

  std::optional<c10::Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(common_device, dy, "gn_nhwc_bwd", "dy");
  c10::impl::check_and_update_common_device(common_device, X, "gn_nhwc_bwd", "X");
  c10::impl::check_and_update_common_device(common_device, weight, "gn_nhwc_bwd", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "gn_nhwc_bwd", "bias");
  c10::impl::check_and_update_common_device(common_device, means, "gn_nhwc_bwd", "means");
  c10::impl::check_and_update_common_device(common_device, rstds, "gn_nhwc_bwd", "rstds");

  c10::OptionalDeviceGuard guard(at::device_of(dy));

  //at::Tensor dy_nhwc = dy.permute({0, 2, 3, 1});
  //at::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  at::Tensor dy_nhwc = dy.permute({0, 2, 1});
  at::Tensor X_nhwc = X.permute({0, 2, 1});
  at::Tensor dX = at::empty_like(X_nhwc);
  at::Tensor dweight = at::empty({C}, X.options());
  at::Tensor dbias = at::empty({C}, X.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    c10::ScalarType::Half,
    c10::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_backward", [&]() {
      run_gn_bwd_kernels<scalar_t>(
      dy_nhwc.const_data_ptr<scalar_t>(), X_nhwc.const_data_ptr<scalar_t>(),
      weight.const_data_ptr<scalar_t>(), bias.const_data_ptr<scalar_t>(),
      means.const_data_ptr<scalar_t>(), rstds.const_data_ptr<scalar_t>(),
      //N, H, W, C, G, act_fn_option,
      N, R, C, G, act_fn_option,
      dX.mutable_data_ptr<scalar_t>(), dweight.mutable_data_ptr<scalar_t>(), dbias.mutable_data_ptr<scalar_t>()
      );
  });
  //return {dX.permute({0, 3, 1, 2}), dweight, dbias};
  return {dX.permute({0, 2, 1}), dweight, dbias};
}

TORCH_LIBRARY(gnop, m) {
  m.def("fwd", &gn_nhwc_fwd);
  m.def("bwd", &gn_nhwc_bwd);
}
