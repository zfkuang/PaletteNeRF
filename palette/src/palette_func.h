#pragma once

#include <stdint.h>
#include <torch/torch.h>

//py::tuple compute_RGB_histogram(py::array_t<float> &colors_rgb, py::array_t<float> &weights, int bits_per_channel);
void rgb_to_hsv(const uint32_t n_rays, const at::Tensor input, at::Tensor output);
void hsv_to_rgb(const uint32_t n_rays, const at::Tensor input, at::Tensor output);