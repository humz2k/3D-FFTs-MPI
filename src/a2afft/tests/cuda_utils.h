#define fftPrecision double

extern void initialize_cuda(fftPrecision** d_myGridCellsBuff1, fftPrecision** d_myGridCellsBuff2, int nlocal);

extern void finalize_cuda(fftPrecision** d_myGridCellsBuff1, fftPrecision** d_myGridCellsBuff2);

extern void util_copy_h2d(fftPrecision** dest, fftPrecision* source, int nlocal);

extern void util_copy_d2h(fftPrecision* dest, fftPrecision** source, int nlocal);