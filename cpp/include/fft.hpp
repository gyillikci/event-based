// Minimal self-contained FFT (no external dependency).
//
// The Python code relies on numpy's rfft/irfft/rfftfreq. To keep the C++ port
// dependency-free (no FFTW), this provides equivalent helpers built on a
// radix-2 Cooley-Tukey FFT with a Bluestein fallback for arbitrary lengths, so
// results match numpy for any N (not just powers of two).
#pragma once

#include <complex>
#include <vector>

namespace ndrone {

using Complex = std::complex<double>;

// In-place FFT of arbitrary length. inverse=true computes the unnormalized
// inverse; call sites divide by N as numpy does.
void fft(std::vector<Complex>& a, bool inverse);

// Real FFT: returns floor(N/2)+1 complex bins, matching numpy.fft.rfft.
std::vector<Complex> rfft(const std::vector<double>& x);

// Inverse real FFT of a spectrum with `n` output samples (numpy.fft.irfft).
std::vector<double> irfft(const std::vector<Complex>& spec, int n);

// Frequencies for an rfft of length n with sample spacing d (numpy.fft.rfftfreq).
std::vector<double> rfftfreq(int n, double d);

// Hann window of length n (numpy.hanning).
std::vector<double> hanning(int n);

}  // namespace ndrone
