#define fftPrecision double
#define cudafftPrecision cufftDoubleComplex
#define doublePrecision

extern void forward_1d_fft(fftPrecision** data, int Ng, int nlocal);

extern void inverse_1d_fft(fftPrecision** data, int Ng, int nlocal);

extern void launch_scale_fft(fftPrecision** data, int Ng, int nlocal, int blockSize);