#include "fft.hpp"

#include <cmath>

namespace ndrone {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool is_power_of_two(size_t n) { return n && ((n & (n - 1)) == 0); }

// Iterative radix-2 Cooley-Tukey FFT (length must be a power of two).
void fft_pow2(std::vector<Complex>& a, bool inverse) {
    const size_t n = a.size();
    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const Complex wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                Complex u = a[i + k];
                Complex v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Bluestein's chirp-z algorithm for arbitrary N, built on the pow2 FFT.
void fft_bluestein(std::vector<Complex>& a, bool inverse) {
    const size_t n = a.size();
    const double sign = inverse ? 1.0 : -1.0;

    size_t m = 1;
    while (m < 2 * n + 1) m <<= 1;

    std::vector<Complex> aa(m, Complex(0, 0));
    std::vector<Complex> bb(m, Complex(0, 0));
    std::vector<Complex> chirp(n);

    for (size_t k = 0; k < n; ++k) {
        // exp(sign * i * pi * k^2 / n)  (use k*k mod 2n to preserve precision).
        double phase = sign * kPi * static_cast<double>((k * k) % (2 * n)) / static_cast<double>(n);
        chirp[k] = Complex(std::cos(phase), std::sin(phase));
        aa[k] = a[k] * chirp[k];
        bb[k] = std::conj(chirp[k]);
        if (k > 0) bb[m - k] = std::conj(chirp[k]);
    }

    fft_pow2(aa, false);
    fft_pow2(bb, false);
    for (size_t i = 0; i < m; ++i) aa[i] *= bb[i];
    fft_pow2(aa, true);
    for (size_t i = 0; i < m; ++i) aa[i] /= static_cast<double>(m);

    for (size_t k = 0; k < n; ++k) a[k] = aa[k] * chirp[k];
}

}  // namespace

void fft(std::vector<Complex>& a, bool inverse) {
    if (a.size() <= 1) return;
    if (is_power_of_two(a.size()))
        fft_pow2(a, inverse);
    else
        fft_bluestein(a, inverse);
}

std::vector<Complex> rfft(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<Complex> a(n);
    for (int i = 0; i < n; ++i) a[i] = Complex(x[i], 0.0);
    fft(a, false);
    a.resize(n / 2 + 1);  // rfft keeps the non-redundant half.
    return a;
}

std::vector<double> irfft(const std::vector<Complex>& spec, int n) {
    // Reconstruct the full Hermitian-symmetric spectrum, then inverse FFT.
    std::vector<Complex> a(n, Complex(0, 0));
    const int half = n / 2 + 1;
    const int limit = std::min(static_cast<int>(spec.size()), half);
    for (int i = 0; i < limit; ++i) a[i] = spec[i];
    for (int i = 1; i < n - i + 1 && i < limit; ++i) {
        a[n - i] = std::conj(spec[i]);
    }
    fft(a, true);
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = a[i].real() / static_cast<double>(n);
    return out;
}

std::vector<double> rfftfreq(int n, double d) {
    const int m = n / 2 + 1;
    std::vector<double> f(m);
    const double scale = 1.0 / (static_cast<double>(n) * d);
    for (int i = 0; i < m; ++i) f[i] = static_cast<double>(i) * scale;
    return f;
}

std::vector<double> hanning(int n) {
    std::vector<double> w(n);
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }
    for (int i = 0; i < n; ++i) {
        w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (n - 1));
    }
    return w;
}

}  // namespace ndrone
