#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "nbd_palette.h"

namespace py = pybind11;

namespace {

template <typename T>
struct pybuffer1D {
public:
    explicit  pybuffer1D(py::array &arr) { set_buffer(arr); }
    explicit  pybuffer1D(py::array_t<T> &arr) { set_buffer(arr); }

    [[nodiscard]] inline T *data() { return static_cast<T *>(m_buf.ptr); }
    [[nodiscard]] inline const T *data() const { return static_cast<const T *>(m_buf.ptr); }
    [[nodiscard]] inline py::ssize_t size() const { return m_buf.size; }

private:
    py::buffer_info m_buf;

    void set_buffer(py::array &arr) {
        if (arr.itemsize() != sizeof(T))
            throw std::runtime_error("itemsize does not match");
        if (arr.ndim() != 1)
            throw std::runtime_error("1D array is expected");

        m_buf = arr.request();
    }
};


inline uint32_t RGB_to_bin_index(const float rgb[3], uint32_t bits_per_channel)
{
    uint32_t index = 0;
    for (int i=0; i<3; ++i) {
        auto c = rgb[i];
        c = std::fmaxf(0.0f, std::fminf(0.999f, c));
        index <<= bits_per_channel;
        index += uint32_t(c * float(1 << bits_per_channel));
    }
    return index;
}

py::tuple compute_RGB_histogram(
        py::array_t<float> &colors_rgb,
        py::array_t<float> &weights,
        int bits_per_channel)
{
    const int bpc = bits_per_channel;
    const int num_bins = 1 << (bpc * 3);
    py::array_t<double> bin_weights{ num_bins };
    py::array_t<float> bin_centers_rgb{ num_bins * 3 };

    pybuffer1D bin_wgt{ bin_weights };
    pybuffer1D bin_cen_rgb{ bin_centers_rgb };
    const pybuffer1D rgb{ colors_rgb };
    const pybuffer1D wgt{ weights };

    for (int i=0; i<num_bins; ++i)
        bin_wgt.data()[i] = 0;

    // compute bin weights
    const auto num_colors = rgb.size() / 3;
    for (int i=0; i<num_colors; ++i) {
        const auto ibin = RGB_to_bin_index(rgb.data() + i * 3, bpc);
        bin_wgt.data()[ibin] += (double)wgt.data()[i];
    }

    // compute the RGB colors at each bin center
    for (int ibin=0; ibin<num_bins; ++ibin) {
        auto code = (uint32_t)ibin;
        for (int i=0; i<3; ++i) {
            const auto c = float(code & ((1 << bpc) - 1)); // lowest bpc bits of the code

            bin_cen_rgb.data()[ibin * 3 + (2 - i)] = (c + 0.5f) / float(1 << bpc);

            code >>= bpc; // use next bpc bits in the following iteration
        }
    }

    bin_centers_rgb.resize({ num_bins, 3 });
    return py::make_tuple(bin_weights, bin_centers_rgb);
}


} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

m.def("compute_RGB_histogram", &::compute_RGB_histogram, R"pbdoc(
    compute the histogram of RGB and weight data
)pbdoc");

} // PYBIND11_MODULE
