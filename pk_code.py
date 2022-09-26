import numpy as np

def calc_power_spectrum(data_k, k_mags = None, k_min = None, ndims = 3, ngrid = 256, box_size = 256, nparticles = 256, print_out = False): 
    
    V = box_size**ndims # volume of the box
    P_k = np.abs(data_k.real)**2 + np.abs(data_k.imag)**2 # this is where we calculate power spectrum!
    
    # now, we need to get the k-values associated with each p(k) value. We will call those "k_mags." You can pass them into this function as a parameter, or calculate them here
    if k_mags is None:  # If you're too lazy to pass this as an argument
        k_vals = np.fft.fftfreq(data_k.shape[0], d = 2*np.pi*box_size/nparticles)
        k_vals_by_dim = np.array(np.meshgrid(*(k_vals,)*ndims ))  # A list containing kx_vals, ky_vals, kz_vals, ..., etc. You can think of these as the k-values in a single dimension
        # For >1 dimension, we use the pythagorean theorem to calculate the k-value associated with each grid cell. We will call those "k_mags"
        k_mags = np.zeros((nparticles,)*ndims)
        for i in range(ndims):
            k_mags += k_vals_by_dim[i]**2
        k_mags = np.sqrt(k_mags)
    
    # now, put everything in bins so that we can take the average (sum and then divide)
    k_mags = k_mags.flatten()
    P_k = P_k.flatten()
    k_min = np.log(np.min(k_mags[k_mags>0]))#-2.5 # just pick something, since np.log(0) is not a thing
    k_max = np.log(np.max(k_mags))   # that might be space e (do I need base 10?)
    k_edges = np.logspace(k_min,k_max,100)   # Try less bins, was 127 before
    #k_edges = np.logspace(np.where(k_min is not None, np.log(k_min), np.log(np.min(k_mags))), np.log(np.max(k_mags)), np.log(int(ngrid / 2))) # I chose to look at 127 bins, but you can do whatever you want
    binned_ks = np.digitize(k_mags, k_edges)
    k_sum = np.bincount(binned_ks, k_mags) # count k's in each bin, weighted by k_mags, so it's like summing the k_mags in each bin
    k_count = np.bincount(binned_ks)       # count k's in each bin (no weighting, just a count)
    P_sum = np.bincount(binned_ks, P_k)    # sum the P_k's in each bin
    if print_out:
        print_stats(P_sum, "P_sum")
        print_stats(k_count, "k_count")
        print(np.where(k_count == 0, True, False))   # If you see nan values, they might come from k_bins with values of 0
    
    # Average! Because the power spectrum is an ensemble average
    k_avg = k_sum / k_count
    P_avg = P_sum / k_count
    
    # return those final values. You can plot these on a simple 2D plot to see your power spectrum
    return P_avg, k_avg