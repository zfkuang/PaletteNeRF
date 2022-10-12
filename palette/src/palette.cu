#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CHANNEL(x) TORCH_CHECK(x <= CHANNEL_MAXIMUM, #x "number of channels is to large")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

#define IS_EQUAL(x, y) (fabs(x-y) < 1e-9)

inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}


template <typename scalar_t>
__global__ void kernal_rgb_to_hsv(
    const uint32_t n_rays, 
    const scalar_t* __restrict__ input, 
    scalar_t* output
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_rays) return;

    // locate 
    input += n * 3;
    output += n * 3;

    scalar_t r = input[0];
    scalar_t g = input[1];
    scalar_t b = input[2];
    
    scalar_t c_max = fmax(fmax(r, g), b);
    scalar_t c_min = fmin(fmin(r, g), b);
    scalar_t diff = c_max - c_min;
    
    scalar_t h = 0, s = 0, v = 0;
    if(IS_EQUAL(diff, 0))
        h = 0;
    else if(IS_EQUAL(c_max, r))
        h = fmod(60 * ((g - b) / diff) + 360, 360);
    else if (IS_EQUAL(c_max, g))
        h = fmod(60 * ((b - r) / diff) + 120, 360);
    else
        h = fmod(60 * ((r - g) / diff) + 240, 360);

    if (IS_EQUAL(c_max, 0))
        s = 0;
    else
        s = (diff / c_max) * 100;

    v = c_max * 100;
    
    output[0] = h; 
    output[1] = s;
    output[2] = v;
}

template <typename scalar_t>
__global__ void kernal_hsv_to_rgb(
    const uint32_t n_rays, 
    const scalar_t* __restrict__ input, 
    scalar_t* output
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_rays) return;

    // locate 
    input += n * 3;
    output += n * 3;

    scalar_t h = input[0];
    scalar_t s = input[1];
    scalar_t v = input[2];
    
    scalar_t c = s / 100 * v / 100;
    scalar_t x = c * (1 - fabs(fmod(h/60, 2)-1));
    scalar_t m = v/100 - c;

    scalar_t r = 0, g = 0, b = 0;
    if(h >= 0 && h < 60){
        r = c,g = x;
    }
    else if(h >= 60 && h < 120){
        r = x,g = c;
    }
    else if(h >= 120 && h < 180){
        g = c,b = x;
    }
    else if(h >= 180 && h < 240){
        g = x,b = c;
    }
    else if(h >= 240 && h < 300){
        r = x,b = c;
    }
    else{
        r = c,b = x;
    }
    

    output[0] = r+m; 
    output[1] = g+m;
    output[2] = b+m;
}

void rgb_to_hsv(uint32_t n_rays, const at::Tensor input, at::Tensor output) {
    static constexpr uint32_t N_THREAD = 128;    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "rgb_to_hsv", ([&] {
        kernal_rgb_to_hsv<<<div_round_up(n_rays, N_THREAD), N_THREAD>>>(n_rays, input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
    }));
}

void hsv_to_rgb(uint32_t n_rays, const at::Tensor input, at::Tensor output) {
    static constexpr uint32_t N_THREAD = 128;    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "hsv_to_rgb", ([&] {
        kernal_hsv_to_rgb<<<div_round_up(n_rays, N_THREAD), N_THREAD>>>(n_rays, input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
    }));
}
