import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import itertools
import numba as nb
from arch import arch_model
import math
from scipy.special import gamma, comb
import pandas as pd
import holidays
from numpy.linalg import solve, inv
from numpy.linalg import cholesky as np_cholesky
import numdifftools as nd
from scipy.linalg import toeplitz, solve_triangular
from scipy.stats import chi2 
from scipy.special import polygamma
from scipy.linalg import cholesky
import statsmodels.api as sm

def pre_whitening(series):
    '''
    Pre-whiten series to remove linear dependencies with an AR(1) plus constant model. 

    r_t = mu + rho * r_{t-1} + e_t

    Then e_t goes into our model and all others (BMSM, GARCH, FIGARCH). Returns the residuals, mu, and rho for use in forecasting.
    '''

    series_array = np.array(series)
    y = series_array[1:]
    x = series_array[:-1]

    ols_model = sm.OLS(y, sm.add_constant(x), hasconst=True)
    results = ols_model.fit()

    mu = results.params[0]
    rho = results.params[1]

    y_hat = results.predict(sm.add_constant(x))
    residuals = y - y_hat

    return residuals, mu, rho

class DataCleaner_SGD:
    '''
    Cleans data from 5 minute intervals with set start and end hours then computes daily Realised Variance
     of log returns ready for forecasting.

    !!Replace US and Singapore holidays with your own country holidays if needed!!
    '''

    def __init__(self, 
                 df: pd.DataFrame, 
                 start_hr: int,
                 end_hr: int, 
                 unit_test: bool) -> None:
        
        '''
        Args:
            - df (pd.DataFrame): DataFrame with datetime index and 'spot' column for closing prices.
            - start_hr (int): Start hour for filtering data.
            - end_hr (int): End hour for filtering data.
            - unit_test (bool): If True, run unit tests to check data cleaning.
        '''

        self.df = df
        self.start_hr = start_hr
        self.end_hr = end_hr
        self.unit_test = unit_test

    def filter_trading_hours(self):
        '''
        Make sure we select the trading hours we are interested in.
        '''

        df_copy = self.df.copy()

        mask = (df_copy.index.time >= pd.Timestamp(f'{self.start_hr:02d}:00:05').time()) & (df_copy.index.time <= pd.Timestamp(f'{self.end_hr:02d}:00:00').time())

        return df_copy[mask]

    def remove_first_day_and_last(self):
        '''
        Remove first and last day incase they dont start at the same time as the rest of the data.
        '''

        df_copy = self.df.copy()

        first_date = df_copy.index.date[0] 
        last_date = df_copy.index.date[-1]

        mask = (df_copy.index.date != first_date) & (df_copy.index.date != last_date)

        return df_copy[mask]

    def check_complete_5M_intervals(self):
        '''
        This function overlays a full 5 minute index onto data and fills missing values to ensure all intervals in
        specified hours are there.
        '''

        df_copy = self.df.copy()

        start_date = df_copy.index.min().floor('D')
        end_date = df_copy.index.max().ceil('D')
        full_index = pd.date_range(start=start_date, end=end_date, freq='5min')

        trading_mask = (full_index.time >= pd.Timestamp(f'{self.start_hr:02d}:00:05').time()) & (full_index.time <= pd.Timestamp(f'{self.end_hr:02d}:00:00').time())
        weekday_mask = full_index.dayofweek < 5 

        trading_index = full_index[weekday_mask & trading_mask]

        df_complete = df_copy.reindex(trading_index).ffill().bfill()  # in case of gaps at start
        
        return df_complete
    
    def unit_test_trading_hours(self):
        '''
        Unit test to check if trading hours are correctly filtered.
        '''
        data_clean = self.df

        print("\nData info:")
        print(f"Start: {data_clean.index.min()}")
        print(f"End: {data_clean.index.max()}")
        print(f"Total observations: {len(data_clean)}")
        print(f"Number of trading days: {len(pd.Series(data_clean.index.date).unique())}")
        print(f"Expected 5 min intervals per day (2-6 is 16 hours but this will vary for diff start/end hours): {16 * 60 / 5}")
        print(f"Expected total observations: {len(pd.Series(data_clean.index.date).unique()) * 16 * 60 / 5}")
        print(f"missing values: {data_clean.isnull().sum().sum()}")

    def get_sum_squared_intraday_returns(self):

        # get returns first
        df_copy = self.df.copy()
        df_copy['spot'] = pd.to_numeric(df_copy['spot'], errors='coerce')
        df_copy['log_spot'] = np.log(df_copy['spot'])
        df_copy['log_returns'] = df_copy['log_spot'].diff().bfill()
        df_copy['squared_log_returns'] = df_copy['log_returns'] ** 2

        squared_r_df = df_copy[['squared_log_returns']]

        idx = pd.Series(squared_r_df.index.date).unique()
        n_days = pd.Series(squared_r_df.index.date).nunique() 
        K = len(squared_r_df) // n_days  # number of returns per day

        realised_variance = np.sum(squared_r_df.values.reshape(n_days, K), axis = 1).reshape(-1, 1)

        res = pd.DataFrame(realised_variance, index=idx, columns=['Realised_Variance'])

        return res, idx
    
    def holiday_dates(self, start, end):

        years = range(start.year, end.year + 1)

        first = holidays.US(years=years)
        second = holidays.Singapore(years=years)

        hols = pd.to_datetime(list(set(first) | set(second))).normalize()
        return pd.DatetimeIndex(hols)
    
    def remove_holidays(self):

        idx  = self.df.index.tz_convert(None)   
        hols = self.holiday_dates(idx.min(), idx.max()) 
        mask = ~idx.normalize().isin(hols)
        return self.df.loc[mask]           


    def clean_data(self):

        self.df = self.filter_trading_hours()
        self.df = self.remove_first_day_and_last()
        self.df = self.check_complete_5M_intervals()
        self.df = self.remove_holidays()

        if self.unit_test:
            self.unit_test_trading_hours()

        self.df, _ = self.get_sum_squared_intraday_returns()

        return self.df
    

class DataCleaner_EUR:
    '''
    Cleans data from 5 minute intervals with set start and end hours then computes daily Realised Variance
     of log returns ready for forecasting.

    !!Replace US and Singapore holidays with your own country holidays if needed!!
    '''

    def __init__(self, 
                 df: pd.DataFrame, 
                 start_hr: int,
                 end_hr: int, 
                 unit_test: bool) -> None:
        
        '''
        Args:
            - df (pd.DataFrame): DataFrame with datetime index and 'spot' column for closing prices.
            - start_hr (int): Start hour for filtering data.
            - end_hr (int): End hour for filtering data.
            - unit_test (bool): If True, run unit tests to check data cleaning.
        '''

        self.df = df
        self.start_hr = start_hr
        self.end_hr = end_hr
        self.unit_test = unit_test

    def filter_trading_hours(self):
        '''
        Make sure we select the trading hours we are interested in.
        '''

        df_copy = self.df.copy()

        mask = (df_copy.index.time >= pd.Timestamp(f'{self.start_hr:02d}:00:05').time()) & (df_copy.index.time <= pd.Timestamp(f'{self.end_hr:02d}:00:00').time())

        return df_copy[mask]

    def remove_first_day_and_last(self):
        '''
        Remove first and last day incase they dont start at the same time as the rest of the data.
        '''

        df_copy = self.df.copy()

        first_date = df_copy.index.date[0] 
        last_date = df_copy.index.date[-1]

        mask = (df_copy.index.date != first_date) & (df_copy.index.date != last_date)

        return df_copy[mask]

    def check_complete_5M_intervals(self):
        '''
        This function overlays a full 5 minute index onto data and fills missing values to ensure all intervals in
        specified hours are there.
        '''

        df_copy = self.df.copy()

        start_date = df_copy.index.min().floor('D')
        end_date = df_copy.index.max().ceil('D')
        full_index = pd.date_range(start=start_date, end=end_date, freq='5min')

        trading_mask = (full_index.time >= pd.Timestamp(f'{self.start_hr:02d}:00:05').time()) & (full_index.time <= pd.Timestamp(f'{self.end_hr:02d}:00:00').time())
        weekday_mask = full_index.dayofweek < 5 

        trading_index = full_index[weekday_mask & trading_mask]

        df_complete = df_copy.reindex(trading_index).ffill().bfill()  # in case of gaps at start
        
        return df_complete
    
    def unit_test_trading_hours(self):
        '''
        Unit test to check if trading hours are correctly filtered.
        '''
        data_clean = self.df

        print("\nData info:")
        print(f"Start: {data_clean.index.min()}")
        print(f"End: {data_clean.index.max()}")
        print(f"Total observations: {len(data_clean)}")
        print(f"Number of trading days: {len(pd.Series(data_clean.index.date).unique())}")
        print(f"Expected 5 min intervals per day (2-6 is 16 hours but this will vary for diff start/end hours): {16 * 60 / 5}")
        print(f"Expected total observations: {len(pd.Series(data_clean.index.date).unique()) * 16 * 60 / 5}")
        print(f"missing values: {data_clean.isnull().sum().sum()}")

    def get_sum_squared_intraday_returns(self):

        # get returns first
        df_copy = self.df.copy()
        df_copy['spot'] = pd.to_numeric(df_copy['spot'], errors='coerce')
        df_copy['log_spot'] = np.log(df_copy['spot'])
        df_copy['log_returns'] = df_copy['log_spot'].diff().bfill()
        df_copy['squared_log_returns'] = df_copy['log_returns'] ** 2

        squared_r_df = df_copy[['squared_log_returns']]

        idx = pd.Series(squared_r_df.index.date).unique()
        n_days = pd.Series(squared_r_df.index.date).nunique() 
        K = len(squared_r_df) // n_days  # number of returns per day

        realised_variance = np.sum(squared_r_df.values.reshape(n_days, K), axis = 1).reshape(-1, 1)

        res = pd.DataFrame(realised_variance, index=idx, columns=['Realised_Variance'])

        return res, idx
    
    def holiday_dates(self, start, end):

        years = range(start.year, end.year + 1)

        first = holidays.US(years=years)
        second = holidays.XECB(years=years)

        hols = pd.to_datetime(list(set(first) | set(second))).normalize()
        return pd.DatetimeIndex(hols)
    
    def remove_holidays(self):

        idx  = self.df.index.tz_convert(None)   
        hols = self.holiday_dates(idx.min(), idx.max()) 
        mask = ~idx.normalize().isin(hols)
        return self.df.loc[mask]           


    def clean_data(self):

        self.df = self.filter_trading_hours()
        self.df = self.remove_first_day_and_last()
        self.df = self.check_complete_5M_intervals()
        self.df = self.remove_holidays()

        if self.unit_test:
            self.unit_test_trading_hours()

        self.df, _ = self.get_sum_squared_intraday_returns()

        return self.df
    

@nb.njit()
def compute_transition_matrix(states, gamma, p_switch = 0.5):
    '''
    We find the transition matrix that is of size 2^kbar x 2^kbar. 
    - states is array and has shape (d, kbar) where d = 2^kbar
    - gamma is array and length kbar
    '''

    d, kbar = states.shape
    T = np.empty((d, d))

    for i in range(d):
        for j in range(d):
            prob = 1

            for k in range(kbar):
                if states[j, k] == states[i, k]: # ie states at t == states at t-1
                    prob *= (1 - gamma[k]) + gamma[k] * p_switch # prob stays + 0.5 * prob switch but we chose this state again
                else:
                    prob *= gamma[k] * p_switch

            T[i, j] = prob

    return T

class BMSM_Forecaster_MLE:

    def __init__(self, 
                  initial_params: list,
                  train_data: np.array, 
                  test_data: np.array,
                  test_realised_variance: np.array,
                  kbar: int,
                  H: int, 
                  B: int,
                  scale: int = 100, 
                  b: float = 2.0,
                  gamma_kbar: float = 0.5,
                  percent_space: bool = True) -> None:
        '''
        Initialise the MSM forecaster with initial parameters and data.
        Args:
            - initial_params (list): Initial parameters for the MSM model.
            - train_data (np.array): Data to be used for estimating. This is the log returns - not raw, not prices, not as %).
            - test_data (np.array): Data to be used for forecasting. This is the log returns - not raw, not prices, not as %).
            - test_realised_variance (np.array): Realised variance for the test data, used for empirical forecasts.
            - kbar (int): Number of states in the MSM model.
            - H (int): Forecast horizon.
            - B (int): Number of draws for posterior distribution.
        '''
        self.scale = scale
        self.percent_space = percent_space
        if percent_space:
            self.train_data = train_data * self.scale
            self.test_data = test_data * self.scale
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.test_realised_variance = test_realised_variance
            self.train_data = train_data
            self.test_data = test_data

        self.initial_params = initial_params
        self.kbar = kbar
        self.H = H
        self.b = b
        self.gamma_kbar = gamma_kbar
        self.B = B

        self.test_size = len(self.test_data) - self.H + 1
        self.predictive_distributions = np.zeros((self.test_size, self.B)) # rows is t and columns is draws - for each row we have B datapoints to compute prob of observing our real forecast

        self.model_params = None
        self.realised_variance_H_forecasts = []
        self.realised_variance_H_true = []
        self.signal_strength = []

        self.optimisation_result = None
        self.cov_theta = None


    def generate_bit_states(self):
        '''
        Build all 2^kbar states as tuples of multipliers m0 and m1=2-m0 in binary form so we can multiply by true m0, m1 later in optimisation. 
        This lists all possible combinations and returns an array with shape (2^kbar, kbar). When kbar is 8 we have 2^kbar = 256.
        '''

        bit_states = np.array([[(i >> k) & 1 for k in range(self.kbar)]
                       for i in range(1 << self.kbar)], dtype=np.uint8)

        return bit_states

    def neg_log_likelihood(self, params, bit_states):
        '''
        This function find the negative log likelihood for each return. We do negative as we minimize rather than maximise.
        '''

        m0, sigma_bar, = params

        gamma1 = 1 - (1 - self.gamma_kbar)**(1 / self.b**(self.kbar-1))
        gamma = np.array([1 - (1 - gamma1)**(self.b**i) for i in range(self.kbar)])

        m1 = 2 - m0
        states = m0*bit_states + m1*(1 - bit_states)
        A = compute_transition_matrix(states, gamma)

        state_mult = states.prod(axis=1)   # \Pi_{i=1}^k M_{i,t}
        d = states.shape[0] 

        Pi = np.ones(d) / d  
        logL = 0.0

        for r in self.train_data:
            Pi_pred = Pi @ A
            scales = sigma_bar * np.sqrt(state_mult)
            density = (1 / (np.sqrt(np.pi * 2) * scales)) * np.exp(- 0.5 * (r / scales) ** 2) 

            pred = np.dot(Pi_pred, density) # inner product
            eps = 1e-300
            logL += np.log(pred + eps)

            numerator = density * Pi_pred # hadamard product
            denominator = np.sum(numerator) + eps
            Pi = numerator / denominator # bayesian updating

        return - logL
    
        
    def estimate_msm_params(self):

        bit_states = self.generate_bit_states()


        bounds = [(0.001, 1.999), 
                        (1e-6, None)] # m0, sigma_bar
        
        result = minimize(
            self.neg_log_likelihood,
            x0 = self.initial_params,
            args = (bit_states,),
            bounds = bounds,
            method = 'L-BFGS-B', 
            options={
            'disp': True,      
            'iprint': 1})
        
        self.optimisation_result = result
        
        if result.success:
            est = result.x
            print(f"Estimated parameters: m0={est[0]:.6e}, sigma_bar={est[1]:.6e}")
        else:
            raise RuntimeError("MSM estimation failed: " + result.message)
        
        final_logL = -result.fun
        print(f"Final log-likelihood: {final_logL:.6e}")

        self.model_params = np.array([est[0], est[1]]) #m0, sigma_bar

        return result, final_logL
    
    def compute_forecasts(self):

        _, _ = self.estimate_msm_params()

        m0, sigma_bar = self.model_params
        gamma1 = 1 - (1 - self.gamma_kbar)**(1 / self.b**(self.kbar-1))
        gamma = np.array([1 - (1 - gamma1)**(self.b**i) for i in range(self.kbar)])
        states = self.generate_bit_states()

        m1 = 2 - m0
        states = m0 * states + m1 * (1 - states)  
        A = compute_transition_matrix(states, gamma)
        state_mult = np.prod(states, axis=1) # \PI_{i=1}^k M_{i,t} ie multiplication
        d = states.shape[0] 
        Pi = np.ones(d) / d 
        scales = sigma_bar * np.sqrt(state_mult)

        # recompute constants from MLE
        for r in self.train_data:
            Pi_pred = Pi @ A
            density = (1/(np.sqrt(2*np.pi)*scales)) * np.exp(-0.5*(r/scales)**2)
            num = Pi_pred * density
            eps = 1e-300
            den = num.sum() + eps
            Pi = num/den

        T = len(self.test_data)

        for t in range(T - self.H + 1):

            current_r2 = 0.0

            for i in range(1, self.H+1):

                Pi_pred_forecasting = Pi @ np.linalg.matrix_power(A, i)

                forecast = np.sum(Pi_pred_forecasting * sigma_bar ** 2 * state_mult) # dot product
                current_r2 += forecast

            self.realised_variance_H_forecasts.append(current_r2)
            current_true_realised = np.sum(self.test_realised_variance[t : t+self.H])
            self.realised_variance_H_true.append(current_true_realised)

            # update Pi
            density = (1 / (np.sqrt(np.pi * 2) * scales)) * np.exp(- 0.5 * (self.test_data[t] / scales) ** 2)
            Pi_pred = Pi @ A
            numerator = density * Pi_pred # hadamard product
            eps = 1e-300
            denominator = np.sum(numerator) + eps
            Pi = numerator / denominator # bayesian updating

        self.realised_variance_H_true = np.array(self.realised_variance_H_true)
        self.realised_variance_H_forecasts = np.array(self.realised_variance_H_forecasts)

        # compute realised volatility
        realised_vol_H_true = np.sqrt(self.realised_variance_H_true)
        realised_vol_H_forecasts = np.sqrt(self.realised_variance_H_forecasts)

        # annualise
        annualiser = np.sqrt(252 / self.H)
        realised_vol_H_true *= annualiser
        realised_vol_H_forecasts *= annualiser

        mse  = np.mean((realised_vol_H_true - realised_vol_H_forecasts)**2)
        mae = np.mean(np.abs(realised_vol_H_true - realised_vol_H_forecasts))
        tss  = np.mean((realised_vol_H_true - realised_vol_H_true.mean())**2)
        R2   = 1 - mse/tss

        naive_forecast = sigma_bar * np.sqrt(252) 
        naive_mse = np.mean((realised_vol_H_true - naive_forecast)**2)
        naive_mae = np.mean(np.abs(realised_vol_H_true - naive_forecast))

        normalised_mse = mse / naive_mse
        normalised_mae = mae / naive_mae

        return normalised_mse, normalised_mae, R2, realised_vol_H_forecasts, realised_vol_H_true

    def compute_hessian_and_cov(self):
        '''
        Computes hessian and covariance matrix using numdifftools as value straight from optimiser is not exactly the hessian from forum online.
        '''

        theta_hat = self.model_params
        bit_states = self.generate_bit_states()  

        def nll(theta):
            return self.neg_log_likelihood(theta, bit_states)

        H = nd.Hessian(nll, method='central')(theta_hat)
        H = 0.5 * (H + H.T) # enforce symmetry so PD
        p = H.shape[0]

        try:
            cholesky(H, lower=True) # check PD
        except Exception:
            H = H + (10 * 1e-8) * np.eye(p) # add reg otherwise

        V = inv(H)
        self.cov_theta = V
        
        return H, V
    
    def param_draws(self, seed = 127):
        '''
        Generate B draws from the posterior distribution of the parameters.
        '''

        _, V = self.compute_hessian_and_cov()

        rng = np.random.default_rng(seed)
        theta_hat = self.model_params

        L = cholesky(V, lower=True)
        draws = theta_hat + (L @ rng.standard_normal(size=(V.shape[0], self.B))).T

        draws[:, 0] = np.clip(draws[:, 0], 0.001, 1.999)  # m0
        draws[:, 1] = np.maximum(draws[:, 1], 1e-8)      # sigma_bar

        return draws
    
    def get_filtered_last_weights(self, theta):
        '''
        Get the Pi, A, state_mult, scales functions from previous train iteration.
        '''
        m0, sigma_bar = theta

        gamma1 = 1 - (1 - self.gamma_kbar)**(1 / self.b**(self.kbar-1))
        gamma = np.array([1 - (1 - gamma1)**(self.b**i) for i in range(self.kbar)])
        states = self.generate_bit_states()

        m1 = 2 - m0
        states = m0 * states + m1 * (1 - states)  
        A = compute_transition_matrix(states, gamma)
        state_mult = np.prod(states, axis=1) # \PI_{i=1}^k M_{i,t} ie multiplication
        d = states.shape[0] 
        Pi = np.ones(d) / d 
        scales = sigma_bar * np.sqrt(state_mult)
        
        for r in self.train_data:
                Pi_pred = Pi @ A
                density = (1/(np.sqrt(2*np.pi)*scales)) * np.exp(-0.5*(r/scales)**2)
                num = Pi_pred * density
                eps = 1e-300
                den = num.sum() + eps
                Pi = num/den

        return Pi, A, state_mult, scales
    
    def compute_predictive_distribution(self):

        draws = self.param_draws()

        Pi_hat, _, _, _ = self.get_filtered_last_weights(self.model_params)
        gamma1 = 1 - (1 - self.gamma_kbar)**(1 / self.b**(self.kbar-1))
        gamma = np.array([1 - (1 - gamma1)**(self.b**i) for i in range(self.kbar)])
        states_int = self.generate_bit_states()

        for b in range(self.B):

            rng = np.random.default_rng(127+b)

            if b % 100 == 0:
                print(f"Processing draw {b+1}/{self.B}")

            params_b = draws[b, :]
            m0, sigma_bar = params_b

            m1 = 2 - m0
            states = m0 * states_int + m1 * (1 - states_int)  #d,k
            A = compute_transition_matrix(states, gamma) #d,d
            A_cdf = np.cumsum(A, axis=1) 
            state_mult = np.prod(states, axis=1) # d
            d = states.shape[0] 
            scales = sigma_bar * np.sqrt(state_mult)

            Pi_t = Pi_hat.copy() # reuse Pi from proper estimation - d,

            T = len(self.test_data)

            for t in range(T - self.H + 1):

                # sample from Pi_t to get index of multiplier states at t-1
                s_idx = int(np.searchsorted(np.cumsum(Pi_t), rng.random()))

                realised_variance_sum = 0.0

                for i in range(1, self.H+1):

                    # propogate state forward ie what state are we in at t+i-1
                    row_cdf = A_cdf[s_idx, :]
                    s_idx = int(np.searchsorted(row_cdf, rng.random()))

                    # realise path
                    sig2 = (sigma_bar**2) * state_mult[s_idx]
                    r = rng.normal() * np.sqrt(sig2)
                    realised_variance_sum += r ** 2
                
                self.predictive_distributions[t, b] = np.sqrt(realised_variance_sum) * np.sqrt(252 / self.H)

                # update Pi
                density = (1 / (np.sqrt(np.pi * 2) * scales)) * np.exp(- 0.5 * (self.test_data[t] / scales) ** 2)
                Pi_pred = Pi_t @ A
                numerator = density * Pi_pred # hadamard product
                eps = 1e-300
                denominator = np.sum(numerator) + eps
                Pi_t = numerator / denominator # bayesian updating

        return self.predictive_distributions
    
class GARCH_Forecaster_MLE:

    def __init__(self, 
                 train_data: np.ndarray,
                 test_data: np.ndarray,
                 test_realised_variance: np.ndarray,
                 initial_params: list = None,
                 p: int = 1,
                 q: int = 1,
                 H: int = 5, 
                 scale: int = 100, 
                 percent_space: bool = True, 
                 B: int = 1000
                 ) -> None:
        
        self.scale = scale
        self.percent_space = percent_space
        if percent_space:
            self.train_data = train_data * self.scale
            self.test_data = test_data * self.scale
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.test_realised_variance = test_realised_variance
            self.train_data = train_data
            self.test_data = test_data

        self.initial_params = initial_params 
        self.p = p
        self.q = q
        self.H = H
        self.B = B

        self.sigma2_prev = None
        self.omega = None
        self.alpha = None
        self.beta = None

        self.predictive_distributions = np.zeros((len(self.test_data) - self.H + 1, self.B))
    
    def garch_filter(self, params):

        omega, alpha, beta = params

        n_train = len(self.train_data)
        sigma2_train = np.zeros(n_train)

        unconditional_var = omega / (1 - alpha - beta)
        
        for t in range(n_train):
            if t == 0:
                sigma2_train[t] = unconditional_var
            else:
                sigma2_train[t] = (omega + 
                                   alpha * self.train_data[t-1]**2 + 
                                   beta * sigma2_train[t-1])

        return sigma2_train
    
    def garch_neg_log_likelihood(self, params):

        _, alpha, beta = params
        
        if alpha + beta >= 1:
            return 1e10  
        
        sigma2 = self.garch_filter(params)
        sigma2 = np.maximum(sigma2, 1e-8)
        self.sigma2_prev = sigma2[-1]
        
        log_L = - np.sum(-np.log(sigma2) - self.train_data**2 / sigma2)
        
        return log_L   
    
    def estimate_garch(self):
        
        bounds = [(1e-8, None),   
                  (1e-8, 0.9999),   
                  (1e-8, 0.9999)]  

        if self.initial_params[1] + self.initial_params[2] >= 1:
            self.initial_params[1] = 0.05
            self.initial_params[2] = 0.90

        self.initial_params = [max(self.initial_params[0], 1e-4), self.initial_params[1], self.initial_params[2]]

        res = minimize(self.garch_neg_log_likelihood,
                        x0=self.initial_params,
                        method='L-BFGS-B',
                        bounds=bounds)
        
        self.omega, self.alpha, self.beta = res.x
        final_log_L = -res.fun

        print(f"Estimated parameters: omega={self.omega}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        print(f"Alpha + Beta = {self.alpha + self.beta:.4f}")
        print(f"Final log-likelihood = {final_log_L:.4f}")

        return self.omega, self.alpha, self.beta, self.sigma2_prev
    
    def forecast_garch(self):

        n_test = len(self.test_data)
        realised_variance_forecasts = np.zeros(n_test - self.H + 1)
        realised_variance_true_H_values = np.zeros(n_test - self.H + 1)
        r_prev = self.train_data[-1]
        sigma2_prev = self.sigma2_prev

        for t in range(n_test - self.H + 1):

            sigma2 = self.omega + self.alpha * (r_prev ** 2) + self.beta * sigma2_prev

            h = np.zeros(self.H)
            h[0] = sigma2

            for i in range(1, self.H):
                h[i] = self.omega + (self.alpha + self.beta) * h[i-1]
                
            realised_variance_forecasts[t] = h.sum()
            realised_variance_true_H_values[t] = np.sum(self.test_realised_variance[t : t + self.H])

            r_prev = self.test_data[t]
            sigma2_prev = sigma2

        realised_volatility_forecasts = np.sqrt(realised_variance_forecasts)
        realised_volatility_true_H_values = np.sqrt(realised_variance_true_H_values)

        # annualise
        annualiser = np.sqrt(252 / self.H)
        realised_volatility_forecasts *= annualiser
        realised_volatility_true_H_values *= annualiser

        mse = np.mean((realised_volatility_forecasts - realised_volatility_true_H_values) ** 2)
        mae = np.mean(np.abs(realised_volatility_forecasts - realised_volatility_true_H_values))
        TSS = np.mean((realised_volatility_true_H_values - realised_volatility_true_H_values.mean()) ** 2)
        R2 = 1 - mse / TSS
        
        naive_forecast = (self.omega / (1 - self.alpha - self.beta)) ** 0.5 * np.sqrt(252) 
        naive_mse = np.mean((realised_volatility_true_H_values - naive_forecast)**2)
        naive_mae = np.mean(np.abs(realised_volatility_true_H_values - naive_forecast))

        normalised_mse = mse / naive_mse
        normalised_mae = mae / naive_mae

        return normalised_mse, normalised_mae, R2, realised_volatility_forecasts, realised_volatility_true_H_values
    
    def compute_hessian_and_cov(self):
        '''
        Computes hessian and covariance matrix using numdifftools as value straight from optimiser is not exactly the hessian from forum online.
        '''

        theta_hat = np.array([self.omega, self.alpha, self.beta]) 

        def nll(theta):
            return self.garch_neg_log_likelihood(theta)

        H = nd.Hessian(nll, method='forward', step=1e-4)(theta_hat)
        H = 0.5 * (H + H.T) # enforce symmetry so PD
        p = H.shape[0]

        try:
            cholesky(H, lower=True) # check PD
        except Exception:
            H = H + (10 * 1e-8) * np.eye(p) # add reg otherwise

        V = inv(H)
        self.cov_theta = V
        
        return H, V
    
    def param_draws(self, seed = 127):
        '''
        Generate B draws from the posterior distribution of the parameters.
        '''

        _, V = self.compute_hessian_and_cov()

        rng = np.random.default_rng(seed)
        theta_hat = np.array([self.omega, self.alpha, self.beta]) 

        L = cholesky(V, lower=True)

        draws = np.empty((0, 3))
        B = self.B

        while draws.shape[0] < B:

            batch = theta_hat + (L @ rng.standard_normal((3, B))).T   

            ok = (batch[:, 0] > 0) & (batch[:, 1] >= 0) & (batch[:, 2] >= 0) \
                & (batch[:, 1] + batch[:, 2] < 1)
            
            draws = np.vstack([draws, batch[ok]])

        return draws[:B]

    def compute_predictive_distribution(self):

        draws = self.param_draws()

        for b in range(self.B):

            rng = np.random.default_rng(127+b)

            if b % 100 == 0:
                print(f"Processing draw {b+1}/{self.B}")

            params_b = draws[b, :]
            omega, alpha, beta = params_b

            r_prev = self.train_data[-1]
            sigma2_prev = self.sigma2_prev

            T = len(self.test_data)

            for t in range(T - self.H + 1):

                sigma2 = omega + alpha * (r_prev ** 2) + beta * sigma2_prev

                h = np.zeros(self.H)
                h[0] = sigma2

                returns = np.zeros(self.H)
                returns[0] = np.sqrt(sigma2) * rng.normal()

                for i in range(1, self.H):
                    h[i] = omega + alpha * returns[i-1] ** 2 + beta * h[i-1]
                    returns[i] = np.sqrt(h[i]) * rng.normal()
                
                realised_variance_sum = np.sum(returns ** 2)
                self.predictive_distributions[t, b] = np.sqrt(realised_variance_sum) * np.sqrt(252 / self.H)

                r_prev = self.test_data[t]
                sigma2_prev = sigma2

        return self.predictive_distributions
    

class FIGARCH_Forecaster:

    def __init__(self, 
                 train_data: np.ndarray,
                 test_data: np.ndarray,
                 test_realised_variance: np.ndarray,
                 initial_params: list,
                 p: int = 1,
                 q: int = 0,
                 H: int = 5, 
                 M: int = 1000, 
                 scale: int = 100,
                 percent_space: bool = True
                 ) -> None:
        
        self.scale = scale
        self.percent_space = percent_space
        if percent_space:
            self.train_data = train_data * self.scale
            self.test_data = test_data * self.scale
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.train_data = train_data
            self.test_data = test_data
            self.test_realised_variance = test_realised_variance

        self.initial_params = initial_params  # [omega, d, beta]
        self.p = p
        self.q = q
        self.H = H
        self.M = M

        self.omega = None
        self.d = None
        self.beta = None
        self.scale = None
        self.sigma2_prev = None

    def fit_figarch_model_library(self):

        gm_t = arch_model(
            self.train_data,
            mean='Zero',
            vol='FIGARCH', p=self.p, q=self.q,
            power = 2.0,
            dist='Normal')
        
        res_t = gm_t.fit(update_freq=0, disp='off')
        print(res_t.summary())
        #res_t.plot()

        #self.scale = res_t.scale
        self.omega = res_t.params['omega'] #/ self.scale ** 2 # also do this for forecasted variances
        self.d = res_t.params['d']
        self.beta = res_t.params['phi']
        self.sigma2_prev = res_t.conditional_volatility[-1]**2


        return self.omega, self.d, self.beta, self.sigma2_prev
    

    def figarch_filter(self, params):

        omega, d, beta = params

        n_train = len(self.train_data)
        sigma2_train = np.zeros(n_train)

        unconditional_var = omega / (1 - beta)

        gamma = np.zeros(self.M+1)
        gamma[0] = 1.0
        for k in range(1, self.M+1):
            gamma[k] = ((k - 1 - d) / k) * gamma[k-1]
        gamma = gamma[1:]
        
        for t in range(n_train):
            if t == 0:
                sigma2_train[t] = unconditional_var
            else:
                recent_returns2 = self.train_data[max(0, t - self.M): t][::-1] ** 2
                fractional_term = gamma[:recent_returns2.shape[0]] @ recent_returns2
                sigma2_train[t] = (omega + 
                                   beta * (sigma2_train[t-1] - self.train_data[t-1]**2)  - 
                                   fractional_term)

        return sigma2_train
    
    def figarch_neg_log_likelihood(self, params):
        
        sigma2 = self.figarch_filter(params)
        sigma2 = np.maximum(sigma2, 1e-8)
        self.sigma2_prev = sigma2[-1]
        
        log_L = - np.sum(-np.log(sigma2) - self.train_data**2 / sigma2)
        
        return log_L  
    
    def estimate_figarch(self):
        
        bounds = [(1e-8, None),   
                  (1e-8, 0.9999),   
                  (1e-8, 0.9999)]  

        self.initial_params = [max(self.initial_params[0], 1e-6), self.initial_params[1], self.initial_params[2]]

        res = minimize(self.figarch_neg_log_likelihood,
                        x0=self.initial_params,
                        method='L-BFGS-B',
                        bounds=bounds)
        
        self.omega, self.d, self.beta = res.x
        final_log_L = -res.fun

        print(f"Estimated parameters: omega={self.omega}, d={self.d:.4f}, beta={self.beta:.4f}")
        print(f"Final log-likelihood = {final_log_L:.4f}")

        return self.omega, self.d, self.beta, self.sigma2_prev

    
    def forecast_figarch(self):

        n_test = len(self.test_data)
        n_train = len(self.train_data)
        realised_variance_forecasts = np.zeros(n_test - self.H + 1)
        realised_variance_true_H_values = np.zeros(n_test - self.H + 1)

        gamma = np.zeros(self.M+1)
        gamma[0] = 1.0
        for k in range(1, self.M+1):
            gamma[k] = ((k - 1 - self.d) / k) * gamma[k-1]
        gamma = gamma[1:]

        full = np.concatenate([self.train_data, self.test_data])
        sigma2_prev = self.sigma2_prev
        r_prev = self.train_data[-1] 

        for t in range(n_test - self.H + 1):
            
            recent_returns2 = full[max(0, n_train + t - self.M): n_train + t][::-1] **2
            fractional_term = gamma[:recent_returns2.shape[0]] @ recent_returns2
            sigma2 = self.omega + self.beta * (sigma2_prev - r_prev**2) - fractional_term

            h = np.zeros(self.H)
            h[0] = sigma2

            copy_recent_returns2 = recent_returns2.copy()

            for i in range(1, self.H):

                past_shock = h[i-1]
                new_recent_returns2 = np.concatenate([[past_shock], copy_recent_returns2[:-1]])
                
                new_fractional_term = gamma[:new_recent_returns2.shape[0]] @ new_recent_returns2
                h[i] = self.omega - new_fractional_term

                copy_recent_returns2 = new_recent_returns2 
                
            realised_variance_forecasts[t] = h.sum()
            realised_variance_true_H_values[t] = np.sum(self.test_realised_variance[t : t + self.H])

            r_prev = self.test_data[t]
            sigma2_prev = sigma2

        realised_volatility_forecasts = np.sqrt(realised_variance_forecasts)
        realised_volatility_true_H_values = np.sqrt(realised_variance_true_H_values)

        # annualise
        annualiser = np.sqrt(252 / self.H)
        realised_volatility_forecasts *= annualiser
        realised_volatility_true_H_values *= annualiser

        mse = np.mean((realised_volatility_forecasts - realised_volatility_true_H_values) ** 2)
        TSS = np.mean((realised_volatility_true_H_values - realised_volatility_true_H_values.mean()) ** 2)
        R2 = 1 - mse / TSS
        print(f"MSE: {mse}, R2: {R2:.6f}")

        return mse, R2, realised_volatility_forecasts, realised_volatility_true_H_values
    

class LMSM_Estimator_GMM:

    def __init__(self, 
                 train_returns: np.ndarray, 
                 initial_params: np.ndarray,
                 lags: list[int] = [1, 5, 10, 20],
                 kbar: int = 10, 
                 b: float = 2.0, 
                 gamma_kbar: float = 0.5, 
                 max_iter: int = 100, 
                 param_tol: float = 1e-6,
                 W_tol: float = 1e-4
                 ) -> None:

        '''
        Parameters:
        train_returns: np.ndarray - array of log returns
        initial_params: np.ndarray - initial parameters for the LMSM, typically [lambda_hat, sigma_bar]
        lags: list[int] - list of lags for the moments
        kbar: int - number of lags in the LMSM
        b: float - parameter for the LMSM
        gamma_kbar: float - parameter for the LMSM
        max_iter: int - maximum number of iterations for the GMM estimation
        param_tol: float - tolerance for convergence of parameters
        W_tol: float - tolerance for convergence of the weighting matrix
        '''
        
        self.train_returns = train_returns
        self.initial_params = initial_params
        self.lags = lags
        self.kbar = kbar
        self.b = b
        self.gamma_kbar = gamma_kbar
        self.max_iter = max_iter
        self.param_tol = param_tol
        self.W_tol = W_tol

        self.W_final = None

        self.gammas = 1 - (1 - self.gamma_kbar) ** (b ** (np.arange(1, kbar + 1) - kbar))

        first_order = -0.6351814227307391             
        second_order =  1.6371559899184157       
        third_order = -4.710737992756197        
        fourth_order = 19.148024309488146 
        self.log_abs_normal_moments = np.array([first_order, second_order, third_order, fourth_order])

        '''
        All of which are computed using the below but we hardcode to save time.
        first_order = 1/2 * (polygamma(0, 1/2) + np.log(2))
        second_order = 1/4 * (polygamma(1, 1/2) + (polygamma(0, 1/2) + np.log(2))**2)
        third_order = 1/8 * (polygamma(2, 1/2) + 3 * polygamma(1, 1/2) * (polygamma(0, 1/2) + np.log(2)) + (polygamma(0, 1/2) + np.log(2))**3)
        fourth_order = 1/16 * (polygamma(3, 1/2) + 4* polygamma(2, 1/2) * (polygamma(0, 1/2) + np.log(2)) + 3 * polygamma(1, 1/2)**2 + 6 * (polygamma(1, 1/2) * (polygamma(0, 1/2) + np.log(2)) **2) + (polygamma(0, 1/2) + np.log(2))**4)
        '''

    def get_theoretical_moments(self, lambda_hat, sigma_hat):
        '''
        Returns theoretical moments in order of moment q=1 and q=2 for each Tlag and finally the variance. We use formulas outlined
        in Lux (2008) only for this (non-RV version of the LMSM). 
        '''

        s2 = 2 * lambda_hat
        theoretical_moments = []

        for Tlag in self.lags:

            interior_sum1 = (1 - (1 - self.gammas) ** Tlag) # (1 - (1 - gamma_i) ** Tlag)
            eta_tpT_T_eta_t_T = - np.sum(interior_sum1 ** 2) * s2  # E[eta_t+T_T, eta_t, T]
            eta_tpT_T_2 = np.sum(interior_sum1) * 2 * s2  # E[eta_t+T, T ** 2]

            k1 = np.sum(interior_sum1 ** 2) * 6 * s2**2 # kappa 1

            # using sum_i,j(i !=j) ai aj = (sum_i ai)^2 - sum_i ai^2
            k2 = (np.sum(interior_sum1) ** 2 - np.sum(interior_sum1 ** 2)) * 4 * s2 ** 2 # kappa 2
            k3 = (np.sum(interior_sum1 ** 2) ** 2 - np.sum(interior_sum1 ** 4)) * 2 * s2 ** 2 # kappa 3
            eta_tpT_T_2_eta_t_T_2 = k1 + k2 + k3  # E[eta_t+T,T **2, eta_t,T ** 2]

            # now in xi space
            xi_tpT_T_xi_t_T = 0.25 * eta_tpT_T_eta_t_T + self.log_abs_normal_moments[0] ** 2 - self.log_abs_normal_moments[1]
            theoretical_moments.append(xi_tpT_T_xi_t_T)

            xi_tpT_T_2_xi_t_T_2 =  0.25 ** 2 * eta_tpT_T_2_eta_t_T_2 - (eta_tpT_T_2 - eta_tpT_T_eta_t_T) \
                                   * (self.log_abs_normal_moments[0] ** 2 - self.log_abs_normal_moments[1]) \
                                   + 3 * self.log_abs_normal_moments[1] ** 2 - 4 * self.log_abs_normal_moments[0] \
                                   * self.log_abs_normal_moments[2] + self.log_abs_normal_moments[3]
            
            theoretical_moments.append(xi_tpT_T_2_xi_t_T_2)

        # append the theoretical variance
        theoretical_moments.append(sigma_hat ** 2)

        theoretical_moments = np.array(theoretical_moments)
        return theoretical_moments
    
    def compute_moment_differences(self, lambda_hat, sigma_hat):
        '''
        Missing values so all moment conditions equal length. 

        Computes the moment differences for the LMSM model - f(phi) function returning the full shape per observation matrix
        (for the weighting matrix) and the expectation (vector version for the objective function).
        '''

        # x_t is our observations as in Lux (2008)
        eps = 1e-12
        log_abs_x_t = np.log(np.abs(self.train_returns) + eps)
        N = len(log_abs_x_t)
        moment_diff_matrix = []
        theoretical_moments = self.get_theoretical_moments(lambda_hat, sigma_hat)

        max_lag = max(self.lags)
        valid_range = range(max_lag, N - max_lag)

        for t in valid_range:
            
            g_t = []

            for idx, Tlag in enumerate(self.lags):

                xi_t_T = log_abs_x_t[t] - log_abs_x_t[t - Tlag]
                xi_tpT_T = log_abs_x_t[t + Tlag] - log_abs_x_t[t]

                moment_1_diff = xi_t_T * xi_tpT_T - theoretical_moments[2 * idx] 
                g_t.append(moment_1_diff)

                moment_2_diff = (xi_t_T ** 2) * (xi_tpT_T ** 2) - theoretical_moments[2 * idx + 1] 
                g_t.append(moment_2_diff)

            # variance term
            moment_var_diff = (self.train_returns[t] ** 2) - theoretical_moments[-1]
            g_t.append(moment_var_diff)

            moment_diff_matrix.append(g_t)

        moment_diff_matrix = np.array(moment_diff_matrix)
        moment_diff_vector = np.mean(moment_diff_matrix, axis=0)  # average over all T

        return moment_diff_matrix, moment_diff_vector

    
    def gmm_objective_function(self, params, W):

        lambda_hat = params[0]
        sigma_hat = params[1]

        _, moment_diff_vector = self.compute_moment_differences(lambda_hat, sigma_hat)

        obj = moment_diff_vector.T @ W @ moment_diff_vector

        return obj

    def compute_hac_covariance(self, G):

        n = G.shape[0]
        L = int(np.floor(4 * (n / 100) ** (2 / 9)))
        weights = np.zeros(L)

        # bartlets weights
        for l in range(1, L + 1):
            weights[l - 1] = 1 - l / (L+1)

        N_0 = G.T @ G / n

        V = N_0.copy()

        for l in range(1, L + 1):

            N_L = G[l:, :].T @ G[:-l, :] / n

            V += weights[l - 1] * (N_L + N_L.T)

        return V

    
    def update_weighting_matrix(self, params):
        '''
        Update by numerically computing covariance matrix of moment diffferences and invert it.
        Use Cholesky decomposition for numerical stability.
        '''

        lamb = params[0] 
        sigma = params[1]  

        moment_diff_matrix, moment_diff_vector = self.compute_moment_differences(lamb, sigma)
        demeaned_matrix = moment_diff_matrix - moment_diff_vector
        Cov = self.compute_hac_covariance(demeaned_matrix) 

        # cholesky
        L = np_cholesky(Cov)  
        I = np.eye(Cov.shape[0])
        L_inv = solve_triangular(L, I, lower=True)
        W_new = L_inv.T @ L_inv 

        return W_new

    def estimate_lmsm(self):

        bounds = [(1e-6, None), (1e-6, None)]
        
        W_curr = np.eye(len(self.lags) * 2 + 1)  # initial weighting matrix
        curr_params = np.array([self.initial_params[0], self.initial_params[1]])

        for i in range(self.max_iter):

            result = minimize(self.gmm_objective_function, 
                              curr_params, 
                              args=(W_curr,), 
                              method='L-BFGS-B',
                              bounds=bounds,
                              options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 1000, 'maxls': 50})

            if result.success:
                new_params = result.x
                new_W = self.update_weighting_matrix(new_params)
            else:
                print("Optimisation failed.")
                print(result.message)
                break

            # test convergence using abs diff of infinty norm - ensure we have elemntwise convergence
            param_err = np.max(np.abs(new_params - curr_params) / np.maximum(1.0, np.maximum(np.abs(new_params), np.abs(curr_params))))
            W_err = np.linalg.norm(new_W - W_curr, 'fro') / max(1.0, np.linalg.norm(W_curr, 'fro'))

            print(f"Iteration {i+1}: param_err={param_err:.6f}, W_err={W_err:.6f}")

            if param_err < self.param_tol and W_err < self.W_tol:
                print(f"Converged after {i+1} iterations with param_err={param_err:.6f}, W_err={W_err:.6f}")
                break

            curr_params = new_params
            W_curr = new_W

        else:
            print(f"Did not converge after {self.max_iter} iterations with param_err={param_err:.6f}, W_err={W_err:.6f}")

        self.final_lambda_hat = new_params[0]
        self.final_sigma_bar = new_params[1]
        print(f"Final estimates: lambda_hat = {self.final_lambda_hat}, sigma_bar = {self.final_sigma_bar:.6f}")

        self.W_final = new_W

        return self.final_lambda_hat, self.final_sigma_bar, result, self.W_final
    
    def wrapper_for_moment_gradient(self, params):

        lambd, sigma_bar = params

        _, moment_diff_vector = self.compute_moment_differences(lambd, sigma_bar)

        return moment_diff_vector

    def get_standard_errors(self):
        '''
        Get standard error sigma / root(n) for parameters which is obtained from diagonal entries of
        asymptotic covariance matrix of the GMM estimator.
        '''

        if self.W_final is None:
            _, _, _, _ = self.estimate_lmsm()

        lambda_hat = self.final_lambda_hat
        sigma_bar = self.final_sigma_bar
        W_final = self.W_final

        gradient_moments = nd.Gradient(self.wrapper_for_moment_gradient)

        final_params = np.array([lambda_hat, sigma_bar])
        gradient_values = gradient_moments(final_params)

        pre_inv_matrix = (gradient_values.T @ W_final @ gradient_values) 

        # invert for variance of parameters
        L = np_cholesky(pre_inv_matrix)  # Cholesky decomposition for numerical stability
        I = np.eye(pre_inv_matrix.shape[0])
        L_inv = solve_triangular(L, I, lower=True)
        variance_of_params = (L_inv.T @ L_inv) 

        standard_errors = np.sqrt(np.diag(variance_of_params)) / np.sqrt(len(self.train_returns))

        return standard_errors
    
    def get_j_prob(self):

        if self.W_final is None:
            _, _, _, _ = self.estimate_lmsm()

        lambda_hat = self.final_lambda_hat
        sigma_bar = self.final_sigma_bar
        W_final = self.W_final

        moment_diff_matrix, moment_diff_vector = self.compute_moment_differences(lambda_hat, sigma_bar)

        j_stat = (moment_diff_vector.T @ W_final @ moment_diff_vector) * moment_diff_matrix.shape[0]

        df = len(moment_diff_vector) - len(self.initial_params)  
        j_p = 1.0 - chi2.cdf(j_stat, df = df)

        return j_stat, j_p

