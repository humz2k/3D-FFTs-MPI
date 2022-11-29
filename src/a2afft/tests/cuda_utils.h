#define fftPrecision double

extern void initialize_cuda(fftPrecision** d_myGridCellsBuff1, fftPrecision** d_myGridCellsBuff2, int nlocal);

extern void finalize_cuda(fftPrecision** d_myGridCellsBuff1, fftPrecision** d_myGridCellsBuff2);