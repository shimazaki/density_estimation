"""Kernel Bandwidth Optimization.
Copyright (c) 2014 Subhasis Ray
License: GPL v3

Original matlab code (sskernel.m) here:
http://2000.jukuin.keio.ac.jp/shimazaki/res/kernel.html

This was translated into Python by Subhasis Ray, NCBS. Tue Jun 10
23:01:43 IST 2014

"""
import numpy as np

def nextpow2(x):
    """Return the smallest intgral power of 2 that >= x"""
    n = 2
    while n < x:
        n = n * 2
    return n

def fftkernel(x, w):
    """y = fftkernel(x,w)
    
    Function `fftkernel' applies the Gauss kernel smoother to an input
    signal using FFT algorithm.
    
    Input argument
    x:    Sample signal vector. 
    w: 	Kernel bandwidth (the standard deviation) in unit of 
    the sampling resolution of x. 
    
    Output argument
    y: 	Smoothed signal.
    
    MAY 5/23, 2012 Author Hideaki Shimazaki
    RIKEN Brain Science Insitute
    http://2000.jukuin.keio.ac.jp/shimazaki

    Ported to Python: Subhasis Ray, NCBS. Tue Jun 10 10:42:38 IST 2014

    """
    L = len(x)
    Lmax = L + 3 * w
    n = nextpow2(Lmax)    
    X = np.fft.fft(x, n)
    f = np.arange(0, n, 1.0)/n
    f = np.concatenate((-f[:n/2], f[n/2:0:-1]))
    K = np.exp(-0.5*(w*2*np.pi*f)**2)
    y = np.fft.ifft(X * K, n)
    y = y[:L].copy()
    return y

def logexp(x):
    if x < 1e2 :
        y = np.log(1 + np.exp(x))
    else:
        y = x
    return y

def ilogexp(x):
    if x < 1e2:
        y = np.log(np.exp(x)-1)
    else:
        y = x
    return y

def cost_function(x, N, w, dt):
    """The cost function 
    Cn(w) = sum_{i,j} int k(x - x_i) k(x - x_j) dx 
    - 2 sum_{i~=j} k(x_i - x_j)"""
    yh = np.abs(fftkernel(x, w / dt))  # density
    #formula for density
    C = np.sum(yh ** 2) * dt - 2 * np.sum(yh * x) * dt + 2 / np.sqrt(2 * np.pi) / w / N 
    C = C * N * N
    # formula for rate
    # C = dt*sum( yh.^2 - 2*yh.*y_hist + 2/sqrt(2*pi)/w*y_hist )
    return C, yh

def sskernel(spiketimes, tin=None, w=None, bootstrap=False):
    """Calculates optimal fixed kernel bandwidth.

    spiketimes: sequence of spike times (sorted to be ascending).
    
    tin: (optional) time points at which the kernel bandwidth is to be estimated. 

    w: (optional) vector of kernel bandwidths. If specified, optimal
    bandwidth is selected from this.

    bootstrap (optional): whether to calculate the 95% confidence
    interval. (default False)

    Returns
    
    A dictionary containing the following key value pairs:
    
    'y': estimated density,
    't': points at which estimation was computed,
    'optw': optimal kernel bandwidth,
    'w': kernel bandwidths examined,
    'C': cost functions of w,
    'confb95': (lower bootstrap confidence level, upper bootstrap confidence level),
    'yb': bootstrap samples.


    Ref: Shimazaki, Hideaki, and Shigeru Shinomoto. 2010. Kernel
    Bandwidth Optimization in Spike Rate Estimation. Journal of
    Computational Neuroscience 29 (1-2):
    171-82. doi:10.1007/s10827-009-0180-4.

    """
    if tin is None:
        time = np.max(spiketimes) - np.min(spiketimes)
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        print dt
        tin = np.linspace(np.min(spiketimes),
                          np.max(spiketimes),
                          min(int(time / dt + 0.5), 1000)) # The 1000 seems somewhat arbitrary
        t = tin
    else:
        time = np.max(tin) - np.min(tin)
        spiketimes = spiketimes[(spiketimes >= np.min(tin)) &
                                (spiketimes <= np.max(tin))].copy()
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        if dt > np.min(np.diff(tin)):
            t = np.linspace(np.min(tin), np.max(tin),
                            min(int(time/dt + 0.5), 1000))
        else:
            t = tin
    dt = np.min(np.diff(tin))
    yhist, bins = np.histogram(spiketimes, np.r_[t - dt/2, t[-1] + dt/2])
    L = len(yhist)
    N = np.sum(yhist)
    yhist = yhist / (N * dt) # density
    optw = None
    y = None
    if w is not None:
        C = np.zeros(len(w))
        Cmin = np.inf
        for k, w_ in enumerate(w):
            C[k], yh = cost_function(yhist, N, w_, dt)
            if C[k] < Cmin:
                Cmin = C[k]
                optw = w_
                y = yh
    else:        
        # Golden section search on a log-exp scale
        wmin = 2 * dt
        wmax = max(spiketimes) - min(spiketimes)
        imax = 20 # max iterations
        w = np.zeros(imax)
        C = np.zeros(imax)
        tolerance = 1e-5
        phi = 0.5 * (np.sqrt(5) + 1) # The Golden ratio
        a = ilogexp(wmin)
        b = ilogexp(wmax)
        c1 = (phi - 1) * a + (2 - phi) * b
        c2 = (2 - phi) * a + (phi - 1) * b
        f1, y1 = cost_function(yhist, N, logexp(c1), dt)
        f2, y2 = cost_function(yhist, N, logexp(c2), dt)
        k = 0
        while (np.abs(b - a) > (tolerance * (np.abs(c1) + np.abs(c2))))\
              and (k < imax):
            if f1 < f2:
                b = c2
                c2 = c1
                c1 = (phi - 1) * a + (2 - phi) * b
                f2 = f1
                f1, y1 = cost_function(yhist, N, logexp(c1), dt)
                w[k] = logexp(c1)
                C[k] = f1
                optw = logexp(c1)
                y = y1 / (np.sum(y1 * dt))
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, y2 = cost_function(yhist, N, logexp(c2), dt)
                w[k] = logexp(c2)
                C[k] = f2
                optw = logexp(c2)
                y = y2 / np.sum(y2 * dt)
            k = k + 1
    # Bootstrap confidence intervals
    confb95 = None
    yb = None
    if bootstrap:
        nbs = 1000
        yb = np.zeros((nbs, len(tin)))
        for ii in range(nbs):
            idx = np.floor(np.random.rand(N) * N).astype(int)
            xb = spiketimes[idx]
            y_histb, bins = np.histogram(xb, np.r_[t - dt/2, t[-1]+dt/2]) / dt / N
            yb_buf = fftkernel(y_histb, optw / dt).real
            yb_buf = yb_buf / np.sum(yb_buf * dt)
            yb[ii,:] = np.interp(tin, t, yb_buf)
        ybsort = np.sort(yb, axis=0)
        y95b = ybsort[np.floor(0.05 * nbs).astype(int), :]
        y95u = ybsort[np.floor(0.95 * nbs).astype(int), :]
        confb95 = (y95b, y95u)
    ret = np.interp(tin, t, y)
    return {'y': ret,
            't': tin,
            'optw': optw,
            'w': w,
            'C': C,
            'confb95': confb95,
            'yb': yb}

from matplotlib import pyplot as plt

def test_sskernel():
    x = 0.5-1*np.log(np.random.rand(1,1e3))
    x = np.random.gamma(5,1,5e2)
    t = np.linspace(0,20,1e3)
    
    ret = sskernel(x,tin=t, bootstrap=True)
    print ret['optw']
    
    y1 = ret['y']
    t = ret['t']
    confb95_l = ret['confb95'][0]
    confb95_u = ret['confb95'][1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, y1, 'r-')

    ax.fill_between(t,confb95_l, confb95_u, facecolor='gray', alpha=0.2)
    
    ymax = np.amax(confb95_u)
    ax.vlines(x.flatten(), 0, 0.05*ymax, colors='k',)
    
    ax.set_ylim([0,1.1*ymax])
    ax.set_ylabel('Probability density')
    
    plt.show()

if __name__ == '__main__':
    test_sskernel()
    
