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
from scipy.linalg import toeplitz, solve_triangular, pinv
from scipy.stats import chi2 
from scipy.special import polygamma
from scipy.linalg import cholesky
import statsmodels.api as sm
import jax.numpy as jnp
import jax

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

        df_complete = df_copy.reindex(trading_index).bfill().ffill()  # in case of gaps at start
        
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

        df_complete = df_copy.reindex(trading_index).bfill().ffill()  # in case of gaps at start
        
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
    
class DataCleaner_HKD:
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

        df_complete = df_copy.reindex(trading_index).bfill().ffill()  # in case of gaps at start
        
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
        second = holidays.HK(years=years)

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
    
class DataCleaner_JPY:
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

        df_complete = df_copy.reindex(trading_index).bfill().ffill()  # in case of gaps at start
        
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
        second = holidays.JP(years=years)

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


def compute_transition_matrix_grad(states, gamma, p_switch=0.5):

    eq = states[None, :, :] == states[:, None, :]        
    stay = (1 - gamma) + gamma * p_switch                 
    switch = gamma * p_switch                            
    probs = jnp.where(eq, stay, switch)

    return jnp.prod(probs, axis=-1)  

class BMSM_Forecaster_MLE_delta_stds:

    def __init__(self, 
                  initial_params: list,
                  train_data: np.array, 
                  test_data: np.array,
                  train_realised_variance: np.array,
                  test_realised_variance: np.array,
                  kbar: int,
                  H: int, 
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
            self.train_realised_variance = train_realised_variance * self.scale ** 2
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.test_realised_variance = test_realised_variance
            self.train_realised_variance = train_realised_variance
            self.train_data = train_data
            self.test_data = test_data

        self.initial_params = initial_params
        self.kbar = kbar
        self.H = H
        self.b = b
        self.gamma_kbar = gamma_kbar

        self.test_size = len(self.test_data) - self.H + 1

        self.model_params = None
        self.realised_variance_H_forecasts_train = []
        self.realised_variance_H_true_train = []
        self.realised_variance_H_forecasts = []
        self.realised_variance_H_true = []
        self.signal_strength = []

        self.optimisation_result = None
        self.cov_theta = None
        self._grad_fn = None


    def generate_bit_states(self):
        '''
        Build all 2^kbar states as tuples of multipliers m0 and m1=2-m0 in binary form so we can multiply by true m0, m1 later in optimisation. 
        This lists all possible combinations and returns an array with shape (2^kbar, kbar). When kbar is 8 we have 2^kbar = 256.
        '''

        bit_states = np.array([[(i >> k) & 1 for k in range(self.kbar)]
                       for i in range(1 << self.kbar)], dtype=np.uint8)

        return bit_states

    def generate_bit_states_grad(self):
        '''
        Build all 2^kbar states as tuples of multipliers m0 and m1=2-m0 in binary form so we can multiply by true m0, m1 later in optimisation. 
        This lists all possible combinations and returns an array with shape (2^kbar, kbar). When kbar is 8 we have 2^kbar = 256.
        '''

        bit_states = jnp.array([[(i >> k) & 1 for k in range(self.kbar)]
                       for i in range(1 << self.kbar)], dtype=jnp.uint8)

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
        self.log_L = final_logL
        print(f"Final log-likelihood: {final_logL:.6e}")

        self.model_params = np.array([est[0], est[1]]) #m0, sigma_bar

        return result, final_logL
    
    def grad_helper_sigma(self, theta, curr_Pi):

        m0, sigma_bar = theta
        gamma1 = 1 - (1 - self.gamma_kbar)**(1 / self.b**(self.kbar-1))
        gamma = jnp.array([1 - (1 - gamma1)**(self.b**i) for i in range(self.kbar)])
        states = self.generate_bit_states_grad()

        m1 = 2 - m0
        states = m0 * states + m1 * (1 - states)  
        A = compute_transition_matrix_grad(states, gamma)
        state_mult = jnp.prod(states, axis=1) # \PI_{i=1}^k M_{i,t} ie multiplication

        total_variance = 0.0

        for i in range(1, self.H + 1):

                Pi_pred_forecasting = curr_Pi @ jnp.linalg.matrix_power(A, i)
                forecast = jnp.sum(Pi_pred_forecasting * sigma_bar ** 2 * state_mult)

                total_variance += forecast

        return total_variance

    def compute_forecasts(self):

        _, _ = self.estimate_msm_params()

        m0, sigma_bar = self.model_params
        gamma1 = 1 - (1 - self.gamma_kbar)**(1 / self.b**(self.kbar-1))
        gamma = np.array([1 - (1 - gamma1)**(self.b**i) for i in range(self.kbar)])
        states = self.generate_bit_states()
        self.bit_states = states
        self.m0 = m0

        m1 = 2 - m0
        states = m0 * states + m1 * (1 - states)  
        A = compute_transition_matrix(states, gamma)
        state_mult = np.prod(states, axis=1) # \PI_{i=1}^k M_{i,t} ie multiplication
        d = states.shape[0] 
        Pi = np.ones(d) / d 
        scales = sigma_bar * np.sqrt(state_mult)

        A_pows = [None] * (self.H + 1)
        A_pows[1] = A.copy()
        for i in range(2, self.H + 1):
            A_pows[i] = A_pows[i-1] @ A

        T_train = len(self.train_data)

        _, Cov = self.compute_hessian_and_cov()
        if self._grad_fn is None:
            self._grad_fn = jax.jit(jax.grad(self.grad_helper_sigma, argnums=0))
        stds_of_sigma_train = np.zeros((T_train - self.H + 1))

        self.Pi_hist_train = np.empty((T_train - self.H + 1, d))

        T = len(self.test_data)
        self.Pi_hist_test  = np.empty((T - self.H + 1, d))

        # recompute constants from MLE and get train forecasts
        for k in range(T_train - self.H + 1):

            current_r2_train = 0.0

            for i in range(1, self.H+1):

                Pi_pred_forecasting_train = Pi @ A_pows[i]

                forecast_train = np.sum(Pi_pred_forecasting_train * sigma_bar ** 2 * state_mult) # dot product
                current_r2_train += forecast_train

            theta_jax = jnp.array([m0, sigma_bar])
            Pi_jax = jnp.array(Pi)
            grad_at_k = self._grad_fn(theta_jax, Pi_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma_train[k] = np.sqrt(var_sigma)

            self.realised_variance_H_forecasts_train.append(current_r2_train)
            current_true_realised_train = np.sum(self.train_realised_variance[k : k+self.H])
            self.realised_variance_H_true_train.append(current_true_realised_train)

            r = self.train_data[k]
            Pi_pred = Pi @ A
            density = (1/(np.sqrt(2*np.pi)*scales)) * np.exp(-0.5*(r/scales)**2)
            num = Pi_pred * density
            eps = 1e-300
            den = num.sum() + eps
            Pi = num/den

            self.Pi_hist_train[k, :] = Pi.copy()

        # continue updating Pi for gap until test set
        for k in range(T_train - self.H + 1, T_train):
            r = self.train_data[k]
            density = (1.0 / (np.sqrt(2.0*np.pi) * scales)) * np.exp(-0.5 * (r / scales)**2)
            Pi_pred = Pi @ A
            num = Pi_pred * density
            den = num.sum() + 1e-300
            Pi = num / den

        
        # annualise
        annualiser = 252 / self.H

        self.realised_variance_H_true_train = np.array(self.realised_variance_H_true_train) * annualiser
        self.realised_variance_H_forecasts_train = np.array(self.realised_variance_H_forecasts_train) * annualiser

        # compute realised volatility
        #realised_vol_H_true_train = np.sqrt(self.realised_variance_H_true_train)
        #realised_vol_H_forecasts_train = np.sqrt(self.realised_variance_H_forecasts_train)

        stds_of_sigma = np.zeros((T - self.H + 1))

        for t in range(T - self.H + 1):

            current_r2 = 0.0

            for i in range(1, self.H+1):

                Pi_pred_forecasting = Pi @ A_pows[i]

                forecast = np.sum(Pi_pred_forecasting * sigma_bar ** 2 * state_mult) # dot product
                current_r2 += forecast

            theta_jax = jnp.array([m0, sigma_bar])
            Pi_jax = jnp.array(Pi)
            grad_at_k = self._grad_fn(theta_jax, Pi_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma[t] = np.sqrt(var_sigma)

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

            self.Pi_hist_test[t, :] = Pi.copy()

        self.realised_variance_H_true = np.array(self.realised_variance_H_true) * annualiser
        self.realised_variance_H_forecasts = np.array(self.realised_variance_H_forecasts) * annualiser

        # VAR SPACE!!
        mse = np.mean((self.realised_variance_H_true - self.realised_variance_H_forecasts)**2)
        mae = np.mean(np.abs(self.realised_variance_H_true - self.realised_variance_H_forecasts))
        tss = np.mean((self.realised_variance_H_true - np.mean(self.realised_variance_H_true))**2)
        R2 = 1 - mse / tss

        naive_forecast = (sigma_bar * np.sqrt(252)) ** 2
        naive_mse = np.mean((self.realised_variance_H_true - naive_forecast)**2)
        naive_mae = np.mean(np.abs(self.realised_variance_H_true - naive_forecast))

        normalised_mse = mse / naive_mse
        normalised_mae = mae / naive_mae

        return normalised_mse, normalised_mae, R2, self.realised_variance_H_forecasts, self.realised_variance_H_true, stds_of_sigma, self.realised_variance_H_forecasts_train, self.realised_variance_H_true_train, stds_of_sigma_train, naive_forecast

    def compute_hessian_and_cov(self):
        '''
        Computes hessian and covariance matrix using numdifftools as value straight from optimiser is not exactly the hessian from forum online.
        '''

        theta_hat = self.model_params
        bit_states = self.generate_bit_states()  

        def nll(theta):
            return self.neg_log_likelihood(theta, bit_states)

        H = nd.Hessian(nll, method='forward', step=1e-4)(theta_hat)
        H = 0.5 * (H + H.T) # enforce symmetry so PD
        p = H.shape[0]

        try:
            cholesky(H, lower=True) # check PD
        except Exception:
            H = H + (10 * 1e-8) * np.eye(p) # add reg otherwise

        V = inv(H)
        self.cov_theta = V
        self.se = np.sqrt(np.diag(V))
        
        return H, V
    
    def access_extra_values(self):

        alpha_p, beta_p, alpha_hat, beta_hat = self.run_mincer_zarnowitz()

        return self.Pi_hist_train, self.Pi_hist_test, self.bit_states, self.m0, self.log_L, self.se, alpha_p, beta_p, alpha_hat, beta_hat

    def run_mincer_zarnowitz(self):

        realised = self.realised_variance_H_true / (self.scale ** 2)
        forecasts = self.realised_variance_H_forecasts / (self.scale ** 2) # convert to decimals for this

        L = int(np.floor(4 * (len(forecasts) / 100) ** (2 / 9)))

        X = sm.add_constant(forecasts) 
        y = realised
        model = sm.OLS(y, X, missing='drop').fit(
            cov_type="HAC",
            cov_kwds={"maxlags": L})
        
        test_alpha = model.t_test("const = 0")
        alpha_t = float(test_alpha.tvalue)
        alpha_p = float(test_alpha.pvalue)

        test_beta  = model.t_test("x1 = 1")
        beta_t = float(test_beta.tvalue)
        beta_p = float(test_beta.pvalue)

        alpha_hat = float(model.params[0])
        beta_hat = float(model.params[1])

        return alpha_p, beta_p, alpha_hat, beta_hat

class GARCH_Forecaster_MLE_1_1_delta_stds:

    def __init__(self, 
                 train_data: np.ndarray,
                 test_data: np.ndarray,
                 train_realised_variance: np.ndarray,
                 test_realised_variance: np.ndarray,
                 initial_params: list = None,
                 H: int = 5, 
                 scale: int = 100, 
                 percent_space: bool = True
                ) -> None:
        
        self.scale = scale
        self.percent_space = percent_space
        if percent_space:
            self.train_data = train_data * self.scale
            self.test_data = test_data * self.scale
            self.train_realised_variance = train_realised_variance * self.scale ** 2
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.test_realised_variance = test_realised_variance
            self.train_realised_variance = train_realised_variance
            self.train_data = train_data
            self.test_data = test_data

        self.initial_params = initial_params 
        self.H = H

        self.sigma2_prev = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self._grad_fn = None
    
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
        
        log_L = - np.sum(- np.log(sigma2) - self.train_data**2 / sigma2) 
        
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
        self.log_L = final_log_L

        print(f"Estimated parameters: omega={self.omega}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        #print(f"Alpha + Beta = {self.alpha + self.beta:.4f}")
        #print(f"Final log-likelihood = {final_log_L:.4f}")

        return self.omega, self.alpha, self.beta, self.sigma2_prev
    
    def grad_helper_sigma(self, theta, r_previous, sigma2_previous):

        omega, alpha, beta = theta

        sigma2 = omega + alpha * (r_previous ** 2) + beta * sigma2_previous

        h = jnp.zeros(self.H)
        h = h.at[0].set(sigma2)
        curr_past_sigma2 = sigma2_previous

        total_variance = sigma2

        for i in range(1, self.H):

            sigma2 = omega + (alpha + beta) * curr_past_sigma2 
            h = h.at[i].set(sigma2)
            curr_past_sigma2 = sigma2

        total_variance = h.sum()

        return total_variance

    def forecast_garch(self):

        n_train = len(self.train_data)

        _, Cov = self.compute_hessian_and_cov()
        if self._grad_fn is None:
            self._grad_fn = jax.jit(jax.grad(self.grad_helper_sigma, argnums=0))
        theta_jax = jnp.array([self.omega, self.alpha, self.beta])
        stds_of_sigma_train = np.zeros((n_train - self.H + 1))

        unconditional_sigma2 = self.omega / (1 - self.alpha - self.beta)
        sigma2_path = np.zeros(n_train + 1)
        sigma2_path[0] = unconditional_sigma2

        for k in range(1, n_train+1):

            sigma2_path[k] = self.omega + self.alpha * (self.train_data[k-1] ** 2) + self.beta * sigma2_path[k-1]

        realised_variance_forecasts_train = np.zeros(n_train - self.H + 1)
        realised_variance_H_true_train = np.zeros(n_train - self.H + 1)

        for k in range(n_train - self.H + 1):

            h = np.zeros(self.H)
            h[0] = sigma2_path[k]

            if k==0:
                r_prev_train = self.train_data[0]
                sigma2_prev_train = sigma2_path[0]
            else:
                r_prev_train = self.train_data[k-1]
                sigma2_prev_train = sigma2_path[k-1]

            for i in range(1, self.H):

                h[i] = self.omega + (self.alpha + self.beta) * h[i-1]

            realised_variance_forecasts_train[k] = h.sum()
            realised_variance_H_true_train[k] = np.sum(self.train_realised_variance[k : k + self.H])

            r_previous_jax = jnp.array(r_prev_train)
            sigma2_previous_jax = jnp.array(sigma2_prev_train)
            grad_at_k = self._grad_fn(theta_jax, r_previous_jax, sigma2_previous_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma_train[k] = np.sqrt(var_sigma)

        annualiser = 252 / self.H
        realised_variance_forecasts_train *= annualiser
        realised_variance_H_true_train *= annualiser

        n_test = len(self.test_data)
        realised_variance_forecasts = np.zeros(n_test - self.H + 1)
        realised_variance_true_H_values = np.zeros(n_test - self.H + 1)
        r_prev = self.train_data[-1]
        sigma2_prev = self.sigma2_prev

        stds_of_sigma = np.zeros((n_test - self.H + 1))

        for t in range(n_test - self.H + 1):

            sigma2 = self.omega + self.alpha * (r_prev ** 2) + self.beta * sigma2_prev

            h = np.zeros(self.H)
            h[0] = sigma2

            for i in range(1, self.H):
                h[i] = self.omega + (self.alpha + self.beta) * h[i-1]
                
            realised_variance_forecasts[t] = h.sum()
            realised_variance_true_H_values[t] = np.sum(self.test_realised_variance[t : t + self.H])

            r_previous_jax = jnp.array(r_prev)
            sigma2_previous_jax = jnp.array(sigma2_prev)
            grad_at_k = self._grad_fn(theta_jax, r_previous_jax, sigma2_previous_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma[t] = np.sqrt(var_sigma)

            r_prev = self.test_data[t]
            sigma2_prev = sigma2

        # annualise
        realised_variance_forecasts *= annualiser
        realised_variance_true_H_values *= annualiser
        self.realised_variance_H_true = realised_variance_true_H_values
        self.realised_variance_H_forecasts = realised_variance_forecasts

        # errors based on variance
        mse  = np.mean((realised_variance_true_H_values - realised_variance_forecasts)**2)
        mae = np.mean(np.abs(realised_variance_true_H_values - realised_variance_forecasts))
        tss  = np.mean((realised_variance_true_H_values - realised_variance_true_H_values.mean())**2)
        R2   = 1 - mse/tss

        sigma_bar = np.std(self.train_data)  
        naive_forecast = (sigma_bar * np.sqrt(252)) ** 2
        naive_mse = np.mean((realised_variance_true_H_values - naive_forecast)**2)
        naive_mae = np.mean(np.abs(realised_variance_true_H_values - naive_forecast))

        normalised_mse = mse / naive_mse
        normalised_mae = mae / naive_mae

        return normalised_mse, normalised_mae, R2, realised_variance_forecasts, realised_variance_true_H_values, stds_of_sigma, realised_variance_forecasts_train, realised_variance_H_true_train, stds_of_sigma_train, naive_forecast

    def compute_hessian_and_cov(self):
        '''
        Computes hessian and covariance matrix using numdifftools as value straight from optimiser is not exactly the hessian from forum online.
        '''

        theta_hat = np.array([self.omega, self.alpha, self.beta]) 

        def nll(theta):
            return self.garch_neg_log_likelihood(theta)

        H = nd.Hessian(nll, method='forward', step=1e-5)(theta_hat)
        H = 0.5 * (H + H.T) # enforce symmetry so PD
        p = H.shape[0]

        try:
            cholesky(H, lower=True) # check PD
        except Exception:
            H = H + (10 * 1e-8) * np.eye(p) # add reg otherwise

        V = inv(H)
        new_V = self._make_psd(V)
        self.cov_theta = new_V
        self.se = np.sqrt(np.diag(new_V))
        
        return H, new_V
    
    def _make_psd(self, M, eps=1e-6):

        M = 0.5 * (M + M.T)
        lamb, S = np.linalg.eigh(M)

        lamb_prime = np.clip(lamb, eps, None)
        B_prime = S @ np.diag(np.sqrt(lamb_prime))

        C = B_prime @ B_prime.T # no normalisation like in paper because we have covariance not correlation
        return C

    def access_extra_values(self):

        alpha_p, beta_p, alpha_hat, beta_hat = self.run_mincer_zarnowitz()

        return self.log_L, self.se, alpha_p, beta_p, alpha_hat, beta_hat

    def run_mincer_zarnowitz(self):

        realised = self.realised_variance_H_true / (self.scale ** 2)
        forecasts = self.realised_variance_H_forecasts / (self.scale ** 2)

        L = int(np.floor(4 * (len(forecasts) / 100) ** (2 / 9)))

        X = sm.add_constant(forecasts) 
        y = realised
        model = sm.OLS(y, X, missing='drop').fit(
            cov_type="HAC",
            cov_kwds={"maxlags": L})
        
        test_alpha = model.t_test("const = 0")
        alpha_t = float(test_alpha.tvalue)
        alpha_p = float(test_alpha.pvalue)

        test_beta  = model.t_test("x1 = 1")
        beta_t = float(test_beta.tvalue)
        beta_p = float(test_beta.pvalue)

        alpha_hat = float(model.params[0])
        beta_hat = float(model.params[1])

        return alpha_p, beta_p, alpha_hat, beta_hat
    
    
class GARCH_Forecaster_MLE_p_1_q_2_delta_stds:

    def __init__(self, 
                 train_data: np.ndarray,
                 test_data: np.ndarray,
                 train_realised_variance: np.ndarray,
                 test_realised_variance: np.ndarray,
                 scale: int = 100,
                 percent_space: bool = True, 
                 H: int = 30
                ) -> None:
        
        self.scale = scale
        self.percent_space = percent_space
        if percent_space:
            self.train_data = train_data * self.scale
            self.test_data = test_data * self.scale
            self.train_realised_variance = train_realised_variance * self.scale ** 2
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.train_data = train_data
            self.test_data = test_data
            self.train_realised_variance = train_realised_variance
            self.test_realised_variance = test_realised_variance

        self.H = H

        self.omega = None
        self.alpha = None
        self.beta1 = None
        self.beta2 = None
        self.sigma2_prev = None

        self._grad_fn = None

    def fit_garch_model_library(self):

        gm_t = arch_model(
            self.train_data,
            mean='Zero',
            vol='GARCH', p=1, q=2,
            power = 2.0,
            dist='Normal', 
            rescale=False)

        res_t = gm_t.fit(update_freq=0, disp='off')
        #print(res_t.summary())
        #res_t.plot()

        params = res_t.params
        self.sigma2_prev = res_t.conditional_volatility[-1]**2 
        self.sigma2_prev2 = res_t.conditional_volatility[-2]**2 

        self.omega = params['omega'] 
        self.alpha = params['alpha[1]']
        self.beta1 = params['beta[1]']
        self.beta2 = params['beta[2]']

        print(f"Estimated parameters: omega={self.omega}, alpha={self.alpha:.4f}, beta1={self.beta1:.4f}, beta2={self.beta2:.4f}")

        self.log_L = res_t.loglikelihood

        return res_t
    
    def forecast_garch(self):

        n_train = len(self.train_data)

        _, Cov = self.compute_hessian_and_cov()

        if self._grad_fn is None:
            self._grad_fn = jax.jit(jax.grad(self.grad_helper_sigma, argnums=0))
        theta_jax = jnp.array([self.omega, self.alpha, self.beta1, self.beta2])
        
        stds_of_sigma_train = np.zeros((n_train - self.H + 1))

        unconditional_sigma2 = max(self.omega / (1 - self.alpha - self.beta1 - self.beta2), 1e-6)
        sigma2_path = np.zeros(n_train + 1)
        sigma2_path[0] = unconditional_sigma2
        sigma2_path[1] = self.omega + self.alpha * (self.train_data[0] ** 2) + self.beta1 * sigma2_path[0] + self.beta2 * unconditional_sigma2

        for k in range(2, n_train + 1):

            sigma2_path[k] = self.omega + self.alpha * (self.train_data[k-1] ** 2) + self.beta1 * sigma2_path[k-1] + self.beta2 * sigma2_path[k-2]

        realised_variance_forecasts_train = np.zeros(n_train - self.H + 1)
        realised_variance_H_true_train = np.zeros(n_train - self.H + 1)

        for k in range(n_train - self.H + 1):

            h = np.zeros(self.H)
            h[0] = sigma2_path[k]

            if k == 0:
                curr_past_sigma2 = unconditional_sigma2

                r_prev_jax = self.train_data[0]
                sigma2_prev_jax = sigma2_path[0]
                sigma2_prev2_jax = sigma2_path[0]

            elif k == 1:

                r_prev_jax = self.train_data[k-1]
                sigma2_prev_jax = sigma2_path[0]
                sigma2_prev2_jax = sigma2_path[0]

                curr_past_sigma2 = sigma2_path[k-1]

            else:
                curr_past_sigma2 = sigma2_path[k-1]

                r_prev_jax = self.train_data[k-1]
                sigma2_prev_jax = sigma2_path[k-1]
                sigma2_prev2_jax = sigma2_path[k-2]

            for i in range(1, self.H):

                h[i] = self.omega + (self.alpha + self.beta1) * h[i-1] + self.beta2 * curr_past_sigma2
                curr_past_sigma2 = h[i-1]

            realised_variance_forecasts_train[k] = h.sum()
            realised_variance_H_true_train[k] = np.sum(self.train_realised_variance[k : k + self.H])

            r_previous_jax = jnp.array(r_prev_jax)
            sigma2_previous_jax = jnp.array(sigma2_prev_jax)
            sigma2_previous2_jax = jnp.array(sigma2_prev2_jax)
            grad_at_k = self._grad_fn(theta_jax, r_previous_jax, sigma2_previous_jax, sigma2_previous2_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma_train[k] = np.sqrt(var_sigma)

        annualiser = 252 / self.H
        realised_variance_forecasts_train *= annualiser
        realised_variance_H_true_train *= annualiser

        n_test = len(self.test_data)
        realised_variance_forecasts = np.zeros(n_test - self.H + 1)
        realised_variance_true_H_values = np.zeros(n_test - self.H + 1)
        r_prev = self.train_data[-1]
        sigma2_prev = self.sigma2_prev
        sigma2_prev2 = self.sigma2_prev2

        stds_of_sigma = np.zeros((n_test - self.H + 1))

        for t in range(n_test - self.H + 1):

            sigma2 = self.omega + self.alpha * (r_prev ** 2) + self.beta1 * sigma2_prev + self.beta2 * sigma2_prev2

            h = np.zeros(self.H)
            h[0] = sigma2
            curr_past_sigma2 = sigma2_prev

            for i in range(1, self.H):

                h[i] = self.omega + (self.alpha + self.beta1) * h[i-1] + self.beta2 * curr_past_sigma2
                curr_past_sigma2 = h[i-1] 

            realised_variance_forecasts[t] = h.sum()
            realised_variance_true_H_values[t] = np.sum(self.test_realised_variance[t : t + self.H])

            r_previous_jax = jnp.array(r_prev)
            sigma2_previous_jax = jnp.array(sigma2_prev)
            sigma2_previous2_jax = jnp.array(sigma2_prev2)
            grad_at_k = self._grad_fn(theta_jax, r_previous_jax, sigma2_previous_jax, sigma2_previous2_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma[t] = np.sqrt(var_sigma)

            r_prev = self.test_data[t]
            sigma2_prev2 = sigma2_prev
            sigma2_prev = sigma2

        # annualise
        annualiser = 252 / self.H
        realised_variance_forecasts *= annualiser
        realised_variance_true_H_values *= annualiser

        self.realised_variance_H_true = realised_variance_true_H_values
        self.realised_variance_H_forecasts = realised_variance_forecasts

        mse  = np.mean((realised_variance_true_H_values - realised_variance_forecasts)**2)
        mae = np.mean(np.abs(realised_variance_true_H_values - realised_variance_forecasts))
        tss  = np.mean((realised_variance_true_H_values - realised_variance_true_H_values.mean())**2)
        R2   = 1 - mse/tss

        sigma_bar = np.std(self.train_data, ddof=1)
        naive_forecast = (sigma_bar * np.sqrt(252)) ** (2)
        naive_mse = np.mean((realised_variance_true_H_values - naive_forecast)**2)
        naive_mae = np.mean(np.abs(realised_variance_true_H_values - naive_forecast))

        normalised_mse = mse / naive_mse
        normalised_mae = mae / naive_mae

        return normalised_mse, normalised_mae, R2, realised_variance_forecasts, realised_variance_true_H_values, stds_of_sigma, realised_variance_forecasts_train, realised_variance_H_true_train, stds_of_sigma_train, naive_forecast


    def grad_helper_sigma(self, theta, r_previous, sigma2_previous, sigma2_previous2):

        omega, alpha, beta1, beta2 = theta

        sigma2 = omega + alpha * (r_previous ** 2) + beta1 * sigma2_previous + beta2 * sigma2_previous2

        h = jnp.zeros(self.H)
        h = h.at[0].set(sigma2)
        curr_past_sigma2 = sigma2_previous
        curr_past_sigma2_2 = sigma2_previous2

        total_variance = sigma2

        for i in range(1, self.H):

            sigma2 = omega + (alpha + beta1) * curr_past_sigma2 + beta2 * curr_past_sigma2_2
            h = h.at[i].set(sigma2)
            curr_past_sigma2, curr_past_sigma2_2 = sigma2, curr_past_sigma2

        total_variance = h.sum()

        return total_variance
    
    def garch_filter(self, params):

        omega, alpha, beta1, beta2 = params

        n_train = len(self.train_data)
        sigma2_train = np.zeros(n_train)

        unconditional_var = omega / (1 - alpha - beta1 - beta2)

        for t in range(n_train):
            if t == 0 or t==1:
                sigma2_train[t] = unconditional_var
            else:
                sigma2_train[t] = (omega + 
                                   alpha * self.train_data[t-1]**2 + 
                                   beta1 * sigma2_train[t-1] + 
                                   beta2 * sigma2_train[t-2])

        return sigma2_train
    
    def garch_neg_log_likelihood(self, params):

        _, alpha, beta1, beta2 = params

        if alpha + beta1 + beta2 >= 1:
            return 1e10

        sigma2 = self.garch_filter(params)
        sigma2 = np.maximum(sigma2, 1e-8)
        self.sigma2_prev = sigma2[-1]
        self.sigma2_prev2 = sigma2[-2]
        
        log_L = - np.sum(-np.log(sigma2) - self.train_data**2 / sigma2)
        
        return log_L   
    
    def compute_hessian_and_cov(self):
        '''
        Computes hessian and covariance matrix using numdifftools as value straight from optimiser is not exactly the hessian from forum online.
        '''

        theta_hat = np.array([self.omega, self.alpha, self.beta1, self.beta2]) 

        def nll(theta):
            return self.garch_neg_log_likelihood(theta)

        H = nd.Hessian(nll, method='forward', step=1e-5)(theta_hat)
        H = 0.5 * (H + H.T) # enforce symmetry so PD
        p = H.shape[0]

        try:
            cholesky(H, lower=True) # check PD
        except Exception:
            H = H + (10 * 1e-6) * np.eye(p) # add reg otherwise

        try:
            V = inv(H)
        except Exception:
            V = pinv(H)

        new_V = self._make_psd(V)
        self.cov_theta = new_V
        self.se = np.sqrt(np.diag(new_V))
        
        return H, new_V

    def _make_psd(self, M, eps=1e-6):

        M = 0.5 * (M + M.T)
        lamb, S = np.linalg.eigh(M)

        lamb_prime = np.clip(lamb, eps, None)
        B_prime = S @ np.diag(np.sqrt(lamb_prime))

        C = B_prime @ B_prime.T # no normalisation like in paper because we have covariance not correlation
        return C
    
    def access_extra_values(self):

        alpha_p, beta_p, alpha_hat, beta_hat = self.run_mincer_zarnowitz()

        return self.log_L, self.se, alpha_p, beta_p, alpha_hat, beta_hat

    def run_mincer_zarnowitz(self):

        realised = self.realised_variance_H_true / (self.scale ** 2)
        forecasts = self.realised_variance_H_forecasts / (self.scale ** 2)

        L = int(np.floor(4 * (len(forecasts) / 100) ** (2 / 9)))

        X = sm.add_constant(forecasts) 
        y = realised
        model = sm.OLS(y, X, missing='drop').fit(
            cov_type="HAC",
            cov_kwds={"maxlags": L})
        
        test_alpha = model.t_test("const = 0")
        alpha_t = float(test_alpha.tvalue)
        alpha_p = float(test_alpha.pvalue)

        test_beta  = model.t_test("x1 = 1")
        beta_t = float(test_beta.tvalue)
        beta_p = float(test_beta.pvalue)

        alpha_hat = float(model.params[0])
        beta_hat = float(model.params[1])

        return alpha_p, beta_p, alpha_hat, beta_hat
    

from scipy.signal import lfilter

class FIGARCH_Forecaster_1_d_0_MLE_delta_stds:

    def __init__(self, 
                 train_data: np.ndarray,
                 test_data: np.ndarray,
                 train_realised_variance: np.ndarray,
                 test_realised_variance: np.ndarray,
                 initial_params: list,
                 H: int = 5, 
                 M: int = 1000, 
                 scale: int = 100,
                 percent_space: bool = True,
                 beta_zero: bool = False # fits the 0,d,0 model for USDHKD without having to rewrite new one
                 ) -> None:
        
        self.scale = scale
        self.percent_space = percent_space
        if percent_space:
            self.train_data = train_data * self.scale
            self.test_data = test_data * self.scale
            self.train_realised_variance = train_realised_variance * self.scale ** 2
            self.test_realised_variance = test_realised_variance * self.scale ** 2
        else:
            self.train_data = train_data
            self.test_data = test_data
            self.train_realised_variance = train_realised_variance
            self.test_realised_variance = test_realised_variance

        self.initial_params = initial_params  # [omega, d, beta]
        self.H = H
        self.M = M
        self.beta_zero = beta_zero

        self.omega = None
        self.d = None
        self.beta = None
        self.sigma2_prev = None
        
        self._gamma_jax = None
        self._grad_fn = None
        self._gamma_cache = {}  
    
    def _compute_gamma_numpy(self, d):

        d_key = round(d, 10) 
        if d_key in self._gamma_cache:
            return self._gamma_cache[d_key]
            
        gamma = np.zeros(self.M+1)
        gamma[0] = 1.0
        for k in range(1, self.M+1):
            gamma[k] = ((k - 1 - d) / k) * gamma[k-1]
        
        gamma_result = gamma[1:]
        self._gamma_cache[d_key] = gamma_result

        return gamma_result
    
    def figarch_filter(self, params):

        if self.beta_zero:
            omega, d = params
            beta = 0.0
        else:
            omega, d, beta = params

        n_train = len(self.train_data)
        sigma2_train = np.zeros(n_train)

        unconditional_var = omega / (1 - beta)

        gamma = self._compute_gamma_numpy(d)
        
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
        
        if not self.beta_zero:

            bounds = [(1e-8, None),   
                    (1e-8, 0.9999),   
                    (1e-8, 0.9999)]  
            
            self.initial_params = [max(self.initial_params[0], 1e-6), self.initial_params[1], self.initial_params[2]]
        else:

            bounds = [(1e-8, None),   
                      (1e-8, 0.9999)]  
            
            self.initial_params = [max(self.initial_params[0], 1e-6), self.initial_params[1]]

        res = minimize(self.figarch_neg_log_likelihood,
                        x0=self.initial_params,
                        method='L-BFGS-B',
                        bounds=bounds)
        
        if self.beta_zero:
            self.omega, self.d = res.x
            self.beta = 0.0
            print(f"Estimated parameters: omega={self.omega}, d={self.d:.4f}")
        else:
            self.omega, self.d, self.beta = res.x
            print(f"Estimated parameters: omega={self.omega}, d={self.d:.4f}, beta={self.beta:.4f}")
        final_log_L = -res.fun

        print(f"Final log-likelihood = {final_log_L:.4f}")

        self.log_L = final_log_L

        return self.omega, self.d, self.beta, self.sigma2_prev
    
    def grad_helper_sigma(self, theta, r_previous, sigma2_previous, recent_returns_squared):

        if self.beta_zero:
            omega, d = theta
            beta = 0.0
        else:
            omega, d, beta = theta

        gamma = jnp.zeros(self.M+1)
        gamma = gamma.at[0].set(1.0)
        for k in range(1, self.M+1):
            gamma = gamma.at[k].set(((k - 1 - d) / k) * gamma[k-1])
        gamma = gamma[1:]

        fractional_term = gamma[:recent_returns_squared.shape[0]] @ recent_returns_squared
        sigma2 = omega + beta * (sigma2_previous - r_previous**2) - fractional_term

        h = jnp.zeros(self.H)
        h = h.at[0].set(sigma2)

        current_returns2 = recent_returns_squared

        for i in range(1, self.H):
            past_shock = h[i-1]
            new_recent_returns2 = jnp.concatenate([jnp.array([past_shock]), current_returns2[:-1]])
            
            new_fractional_term = gamma[:new_recent_returns2.shape[0]] @ new_recent_returns2
            h = h.at[i].set(omega - new_fractional_term)
            
            current_returns2 = new_recent_returns2

        total_variance = h.sum()

        return total_variance

    def forecast_figarch(self):

        n_train = len(self.train_data)

        _, Cov = self.compute_hessian_and_cov()
        if self._grad_fn is None:
            self._grad_fn = jax.jit(jax.grad(self.grad_helper_sigma, argnums=0))
        stds_of_sigma_train = np.zeros((n_train - self.H + 1))

        if self.beta_zero:
            theta_jax = jnp.array([self.omega, self.d])
        else:
            theta_jax = jnp.array([self.omega, self.d, self.beta])

        gamma = self._compute_gamma_numpy(self.d)         
        r2_train = self.train_data ** 2

        frac_full_train = lfilter(gamma, [1.0], r2_train)
        frac_train = np.concatenate(([0.0], frac_full_train[:-1]))

        sigma2_path = np.zeros(n_train + 1)
        unconditional_sigma2 = self.omega / (1 - self.beta)
        sigma2_path[0] = unconditional_sigma2

        for t in range(1, n_train + 1):

            sigma2_path[t] = (
                self.omega
                + self.beta * (sigma2_path[t - 1] - r2_train[t - 1])
                - frac_train[t - 1])

        realised_variance_forecasts_train = np.zeros(n_train - self.H + 1)
        realised_variance_H_true_train    = np.zeros(n_train - self.H + 1)

        for k in range(n_train - self.H + 1):

            r2_win = r2_train[max(0, k - self.M): k]

            if r2_win.size < self.M:

                pad = np.zeros(self.M - r2_win.size, dtype=r2_win.dtype)
                r2_win = np.concatenate([pad, r2_win])

            recent_returns2 = r2_win[::-1]  

            h = np.zeros(self.H)
            h[0] = sigma2_path[k]
            current_returns2 = recent_returns2.copy()  

            if k == 0:
                r_prev_jax = self.train_data[0]
                sigma_2_previous_jax = sigma2_path[0]
            else:
                r_prev_jax = self.train_data[k - 1]
                sigma_2_previous_jax = sigma2_path[k - 1]

            for i in range(1, self.H):
                past_shock = h[i-1]

                current_returns2 = np.concatenate(([past_shock], current_returns2[:-1]))
                new_fractional_term = gamma @ current_returns2[:gamma.shape[0]]
                h[i] = self.omega - new_fractional_term

            r_previous_jax = jnp.array(r_prev_jax)
            sigma2_previous_jax = jnp.array(sigma_2_previous_jax)
            recent_returns_squared_jax = jnp.array(current_returns2)

            grad_at_k = self._grad_fn(theta_jax, r_previous_jax, sigma2_previous_jax, recent_returns_squared_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma_train[k] = np.sqrt(var_sigma)

            realised_variance_forecasts_train[k] = h.sum()
            realised_variance_H_true_train[k] = np.sum(self.train_realised_variance[k : k + self.H])


        annualiser = 252 / self.H
        realised_variance_forecasts_train *= annualiser
        realised_variance_H_true_train *= annualiser


        n_test = len(self.test_data)
        realised_variance_forecasts = np.zeros(n_test - self.H + 1)
        realised_variance_true_H_values = np.zeros(n_test - self.H + 1)

        full = np.concatenate([self.train_data, self.test_data])
        r2_full = full ** 2

        frac_full_all = lfilter(gamma, [1.0], r2_full)
        frac_all = np.concatenate(([0.0], frac_full_all[:-1]))

        sigma2_prev = self.sigma2_prev
        r_prev = self.train_data[-1]

        stds_of_sigma = np.zeros((n_test - self.H + 1))

        for t in range(n_test - self.H + 1):
            idx = n_train + t
            fractional_term = frac_all[idx]  
            sigma2 = self.omega + self.beta * (sigma2_prev - r_prev**2) - fractional_term

            h = np.zeros(self.H)
            h[0] = sigma2

            r2_win = r2_full[max(0, idx - self.M): idx]

            if r2_win.size < self.M:

                pad = np.zeros(self.M - r2_win.size, dtype=r2_win.dtype)
                r2_win = np.concatenate([pad, r2_win])

            current_returns2 = r2_win[::-1]  

            for i in range(1, self.H):
                past_shock = h[i-1]
                current_returns2 = np.concatenate([[past_shock], current_returns2[:-1]])
                new_fractional_term = gamma @ current_returns2[:gamma.shape[0]]
                h[i] = self.omega - new_fractional_term

            realised_variance_forecasts[t] = h.sum()
            realised_variance_true_H_values[t] = np.sum(self.test_realised_variance[t : t + self.H])

            r_previous_jax = jnp.array(r_prev)
            sigma2_previous_jax = jnp.array(sigma2_prev)
            recent_returns_squared_jax = jnp.array(current_returns2)
            grad_at_k = self._grad_fn(theta_jax, r_previous_jax, sigma2_previous_jax, recent_returns_squared_jax)
            var_sigma = grad_at_k @ Cov @ grad_at_k.T
            stds_of_sigma[t] = np.sqrt(var_sigma)

            r_prev = self.test_data[t]
            sigma2_prev = sigma2

        # annualise
        realised_variance_forecasts *= annualiser
        realised_variance_true_H_values *= annualiser

        self.realised_variance_H_true = realised_variance_true_H_values
        self.realised_variance_H_forecasts = realised_variance_forecasts

        mse = np.mean((realised_variance_forecasts - realised_variance_true_H_values) ** 2)
        mae = np.mean(np.abs(realised_variance_forecasts - realised_variance_true_H_values))
        TSS = np.mean((realised_variance_true_H_values - realised_variance_true_H_values.mean()) ** 2)
        R2 = 1 - mse / TSS
        
        #naive_forecast = (self.omega / (1 - self.beta)) ** 0.5 * np.sqrt(252) 
        naive_forecast = (np.std(self.train_data) * np.sqrt(252))**2
        naive_mse = np.mean((realised_variance_true_H_values - naive_forecast)**2)
        naive_mae = np.mean(np.abs(realised_variance_true_H_values - naive_forecast))

        normalised_mse = mse / naive_mse
        normalised_mae = mae / naive_mae

        return normalised_mse, normalised_mae, R2, realised_variance_forecasts, realised_variance_true_H_values, stds_of_sigma, realised_variance_forecasts_train, realised_variance_H_true_train, stds_of_sigma_train, naive_forecast

    def compute_hessian_and_cov(self):
        '''
        Computes hessian and covariance matrix using numdifftools with optimized settings.
        '''

        if self.beta_zero:
            theta_hat = np.array([self.omega, self.d])
        else:
            theta_hat = np.array([self.omega, self.d, self.beta])

        def nll(theta):
            return self.figarch_neg_log_likelihood(theta)

        H = nd.Hessian(nll, method='central', step=1e-5)(theta_hat)
        H = 0.5 * (H + H.T) # enforce symmetry so PD
        p = H.shape[0]

        try:
            cholesky(H, lower=True) # check PD
        except Exception:
            H = H + (10 * 1e-8) * np.eye(p) # add reg otherwise

        V = inv(H)
        self.cov_theta = V
        self.se = np.sqrt(np.diag(V))

        return H, V
    
    def access_extra_values(self):

        alpha_p, beta_p, alpha_hat, beta_hat = self.run_mincer_zarnowitz()

        return self.log_L, self.se, alpha_p, beta_p, alpha_hat, beta_hat

    def run_mincer_zarnowitz(self):

        realised = self.realised_variance_H_true / (self.scale ** 2)
        forecasts = self.realised_variance_H_forecasts / (self.scale ** 2)

        L = int(np.floor(4 * (len(forecasts) / 100) ** (2 / 9)))

        X = sm.add_constant(forecasts) 
        y = realised
        model = sm.OLS(y, X, missing='drop').fit(
            cov_type="HAC",
            cov_kwds={"maxlags": L})
        
        test_alpha = model.t_test("const = 0")
        alpha_t = float(test_alpha.tvalue)
        alpha_p = float(test_alpha.pvalue)

        test_beta  = model.t_test("x1 = 1")
        beta_t = float(test_beta.tvalue)
        beta_p = float(test_beta.pvalue)

        alpha_hat = float(model.params[0])
        beta_hat = float(model.params[1])

        return alpha_p, beta_p, alpha_hat, beta_hat
    