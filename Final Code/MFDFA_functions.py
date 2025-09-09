import numpy as np
from pykalman import KalmanFilter

def get_segments1(s, N, Y):
    '''
    1 here means we get segments excluding left over parts at the end. By constrast, 2 is for left over parts at start are removed.
    '''

    Ns = N // s
    segments = list()
    for i in range(Ns):
        curr_seg = Y[i*s : (i+1) * s]
        segments.append(curr_seg)

    return segments

def get_segments2(s, N, Y):

    Ns = N // s
    buffer = N - Ns * s

    segments = list()
    for i in range(Ns):
        curr_seg = Y[buffer + i*s : buffer + (i+1) * s]
        segments.append(curr_seg)

    return segments

def poly_fit_and_variance1(segments, s, m, N):

    Ns = N // s

    F = list()
    fits = list()

    for i in range(Ns):

        seg = segments[i]
        start_idx = i * s
        end_idx = (i+1) * s

        x = np.arange(start_idx, end_idx, 1)
        poly_fit_coeff = np.polyfit(x, seg, deg = m)
        poly_fit = np.poly1d(poly_fit_coeff)
        y_poly = poly_fit(x)
        fits.append(y_poly)

        F_curr = (1 / s) * np.sum((seg - y_poly) ** 2)
        F.append(F_curr)

    # returns variance of each segment for given s as a list of all segments
    return F, fits

def poly_fit_and_variance2(segments, s, m, N):

    Ns = N // s
    buffer = N - Ns * s

    F = np.zeros(Ns)
    fits = list()

    for i in range(Ns):

        seg = segments[i]
        start_idx = buffer + i * s
        end_idx = buffer + (i+1) * s

        x = np.arange(start_idx, end_idx, 1)
        poly_fit_coeff = np.polyfit(x, seg, deg = m)
        poly_fit = np.poly1d(poly_fit_coeff)
        y_poly = poly_fit(x)
        fits.append(y_poly)

        F[i] = (1 / s) * np.sum((seg - y_poly) ** 2) # actually F^2

    # returns variance of each segment for given s as a list of all segments
    return F, fits, buffer

def moment_fluctuations(F1, F2, N, s, q):

    Ns = N //s 

    F = np.concatenate((F1, F2))

    Ns_total = 2 * Ns

    if q == 0:
        F_q = np.exp((1 / (2 * Ns_total)) * np.sum(np.log(F)))
    else:
        F_q = ((1 / Ns_total) * np.sum(F ** (q / 2))) ** (1/q)

    return F_q

def get_moment_fluctuations_q(q, s_min, s_max, Y, m, numb_ss):
    N = len(Y)

    ss = np.logspace(np.log10(s_min), np.log10(s_max), num=numb_ss)
    ss = np.unique(np.round(ss).astype(int))

    F_qs = np.zeros(len(ss))

    for i in range(len(ss)):
        seg1 = get_segments1(s=ss[i], N = N, Y=Y)
        seg2 = get_segments2(s = ss[i], N=N, Y=Y)

        F1, fits1 = poly_fit_and_variance1(segments=seg1, s = ss[i], m = m, N= N)
        F2, fits2, buffer = poly_fit_and_variance2(segments=seg2, s = ss[i], m = m, N= N)

        F_qs[i] = moment_fluctuations(F1 = F1, F2=F2, N = N, s = ss[i], q = q)

    return F_qs, ss, fits1, fits2, buffer

def get_generalised_hurst(F_q, ss):

    mask = F_q > 0

    #ss = ss[:, np.newaxis]

    eps = np.finfo(float).tiny      # ≈ 2.2 × 10⁻³⁰⁸
    F_q_safe = np.where(F_q <= 0, eps, F_q)
    log_Fq   = np.log(F_q_safe)
        
    log_s = np.log(ss)
    #log_Fq = np.log(F_q)

    slope, _ = np.polyfit(log_s, log_Fq, 1)

    return slope

def get_mf_spectrum(hq, qs):
    '''
    This must be edited depending on where h(1) is ie the qs array.
    '''
    idx = np.where(qs == 1)[0][0]
    #tao_q = qs * hq - 1 - qs * (hq[idx] - 1)
    tao_q = qs * hq - 1  
    '''
    As given by The origins of multifractality in financial time series
    and the eﬀect of extreme events - made adjustments - 
    WHICH ONE??'''

    alpha_q = np.gradient(tao_q, qs)

    f_alpha = alpha_q * qs - tao_q

    return tao_q, alpha_q, f_alpha


def poly_fit_and_variance_kalman(segments, s, N):
    '''
    Does same as poly fit but with kalman - we do not need buffer here as no x index algebra so only one func. 
    '''

    Ns = N // s
    buffer = N - Ns * s

    F = list()
    fits = list()

    for i in range(Ns):

        seg = segments[i]

        P0 = seg.var()
        Q = 0.06 # how much smoothing we apply, towards 1 we apply more

        kf = KalmanFilter(transition_matrices=[1],              # Assumes random walk
                        observation_matrices=[1],             # No data transformations
                        initial_state_mean=seg[0], # Initial estimate i.e. actual observations
                        initial_state_covariance=P0,          # Initial uncertainty proxy
                        observation_covariance=5,             # Measurement of trust in our observations
                        transition_covariance=Q)           # Proxy for data smoothness
        
        ## All very arbitrary params from lecture notebook ##

        # Fit model and make predictions
        state_means, _ = kf.filter(seg)
        state_means = state_means.flatten()
        y_poly = state_means

        fits.append(y_poly)

        F_curr = (1 / s) * np.sum((seg - y_poly) ** 2)
        F.append(F_curr)

    # returns variance of each segment for given s as a list of all segments
    return F, fits, buffer

def get_moment_fluctuations_q_kalman(q, s_min, s_max, Y, numb_ss):
    N = len(Y)

    ss = np.logspace(np.log10(s_min), np.log10(s_max), num=numb_ss)
    ss = np.unique(np.round(ss).astype(int))

    F_qs = np.zeros(len(ss))

    for i in range(len(ss)):
        seg1 = get_segments1(s=ss[i], N = N, Y=Y)
        seg2 = get_segments2(s = ss[i], N=N, Y=Y)

        F1, fits1, _ = poly_fit_and_variance_kalman(segments=seg1, s = ss[i], N= N)
        F2, fits2, buffer = poly_fit_and_variance_kalman(segments=seg2, s = ss[i], N= N)

        F_qs[i] = moment_fluctuations(F1 = F1, F2=F2, N = N, s = ss[i], q = q)

    return F_qs, ss, fits1, fits2, buffer
