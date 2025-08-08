import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
from scipy.stats import norm

class Clean_Implied_Vols_SGD:
    '''
    Use to clean implied vol data taken from Refintiv should return bid, ask, mids, all in correct order, remove nans
    and align with previous data dates. 

    Again change as needed for holidays in specific countries.
    '''

    def __init__(self, 
                data: pd.DataFrame, 
                start_date: pd.Timestamp, 
                end_date: pd.Timestamp, 
                align_df1: pd.DataFrame, # RV and daily data to align new data with
                align_df2: pd.DataFrame 
                ) -> None:
        
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.align_df1 = align_df1
        self.align_df2 = align_df2

        self.clean_data = None

    def holiday_dates(self):

        years = range(self.start_date.year, self.end_date.year + 1)

        first = holidays.US(years=years)
        second = holidays.Singapore(years=years)

        hols = pd.to_datetime(list(set(first) | set(second))).normalize()
        return pd.DatetimeIndex(hols)
    
    def remove_holidays(self):

        idx  = self.clean_data['Exchange Date']
        hols = self.holiday_dates() 
        mask = ~idx.isin(hols)
        return self.clean_data.loc[mask] 
    
    def align_datasets(self):

        df1_dates_aligned = pd.to_datetime(self.align_df1.index).normalize()
        df2_dates_aligned = pd.to_datetime(self.align_df2.index).normalize()
        iv_dates_aligned = pd.to_datetime(self.clean_data['Exchange Date']).dt.normalize()

        common_dates = df1_dates_aligned.intersection(iv_dates_aligned).intersection(df2_dates_aligned)

        df1 = self.align_df1[pd.to_datetime(self.align_df1.index).normalize().isin(common_dates)]
        df2 = self.align_df2[pd.to_datetime(self.align_df2.index).normalize().isin(common_dates)]
        self.clean_data = self.clean_data[self.clean_data['Exchange Date'].dt.normalize().isin(common_dates)]

        return df1, df2, self.clean_data

    def get_clean_data(self):

        self.clean_data = self.data.copy()
        self.clean_data = self.clean_data[::-1]
        self.clean_data['Exchange Date'] = pd.to_datetime(self.clean_data['Exchange Date'])
        self.clean_data['Bid'] = pd.to_numeric(self.clean_data['Bid'], errors='coerce')
        self.clean_data['Ask'] = pd.to_numeric(self.clean_data['Ask'], errors='coerce')
        self.clean_data.dropna(inplace=True)

        mask = (self.clean_data['Exchange Date'] >= self.start_date) & (self.clean_data['Exchange Date'] <= self.end_date)
        self.clean_data = self.clean_data[mask]

        self.clean_data['Mid'] = (self.clean_data['Bid'] + self.clean_data['Ask']) / 2
        self.clean_data = self.clean_data.drop(columns=['BidNet'])

        self.clean_data = self.remove_holidays()

        df1, df2, _ = self.align_datasets()

        return self.clean_data, df1, df2
    
class Clean_Implied_Vols_EUR:
    '''
    Use to clean implied vol data taken from Refintiv should return bid, ask, mids, all in correct order, remove nans
    and align with previous data dates. 

    Again change as needed for holidays in specific countries.
    '''

    def __init__(self, 
                data: pd.DataFrame, 
                start_date: pd.Timestamp, 
                end_date: pd.Timestamp, 
                align_df1: pd.DataFrame, # RV and daily data to align new data with
                align_df2: pd.DataFrame 
                ) -> None:
        
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.align_df1 = align_df1
        self.align_df2 = align_df2

        self.clean_data = None

    def holiday_dates(self):

        years = range(self.start_date.year, self.end_date.year + 1)

        first = holidays.US(years=years)
        second = holidays.XECB(years=years)

        hols = pd.to_datetime(list(set(first) | set(second))).normalize()
        return pd.DatetimeIndex(hols)
    
    def remove_holidays(self):

        idx  = self.clean_data['Exchange Date']
        hols = self.holiday_dates() 
        mask = ~idx.isin(hols)
        return self.clean_data.loc[mask] 
    
    def align_datasets(self):

        df1_dates_aligned = pd.to_datetime(self.align_df1.index).normalize()
        df2_dates_aligned = pd.to_datetime(self.align_df2.index).normalize()
        iv_dates_aligned = pd.to_datetime(self.clean_data['Exchange Date']).dt.normalize()

        common_dates = df1_dates_aligned.intersection(iv_dates_aligned).intersection(df2_dates_aligned)

        df1 = self.align_df1[pd.to_datetime(self.align_df1.index).normalize().isin(common_dates)]
        df2 = self.align_df2[pd.to_datetime(self.align_df2.index).normalize().isin(common_dates)]
        self.clean_data = self.clean_data[self.clean_data['Exchange Date'].dt.normalize().isin(common_dates)]

        return df1, df2, self.clean_data

    def get_clean_data(self):

        self.clean_data = self.data.copy()
        self.clean_data = self.clean_data[::-1]
        self.clean_data['Exchange Date'] = pd.to_datetime(self.clean_data['Exchange Date'])
        self.clean_data['Bid'] = pd.to_numeric(self.clean_data['Bid'], errors='coerce')
        self.clean_data['Ask'] = pd.to_numeric(self.clean_data['Ask'], errors='coerce')
        self.clean_data.dropna(inplace=True)

        mask = (self.clean_data['Exchange Date'] >= self.start_date) & (self.clean_data['Exchange Date'] <= self.end_date)
        self.clean_data = self.clean_data[mask]

        self.clean_data['Mid'] = (self.clean_data['Bid'] + self.clean_data['Ask']) / 2
        self.clean_data = self.clean_data.drop(columns=['BidNet'])

        self.clean_data = self.remove_holidays()

        df1, df2, _ = self.align_datasets()

        return self.clean_data, df1, df2
    
class Clean_Implied_Vols_EUR_with_smile:
    '''
    Use to clean implied vol data taken from Refintiv should return bid, ask, mids, all in correct order, remove nans
    and align with previous data dates. 

    Again change as needed for holidays in specific countries.
    '''

    def __init__(self, 
                data: pd.DataFrame, 
                start_date: pd.Timestamp, 
                end_date: pd.Timestamp, 
                align_df1: pd.DataFrame, # RV and daily data to align new data with
                align_df2: pd.DataFrame, 
                smile_df: pd.DataFrame # align smile data as well
                ) -> None:
        
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.align_df1 = align_df1
        self.align_df2 = align_df2
        self.smile_df = smile_df

        self.clean_data = None

    def holiday_dates(self):

        years = range(self.start_date.year, self.end_date.year + 1)

        first = holidays.US(years=years)
        second = holidays.XECB(years=years)

        hols = pd.to_datetime(list(set(first) | set(second))).normalize()
        return pd.DatetimeIndex(hols)
    
    def remove_holidays(self):

        idx  = self.clean_data['Exchange Date']
        hols = self.holiday_dates() 
        mask = ~idx.isin(hols)
        return self.clean_data.loc[mask] 
    
    def align_datasets(self):

        df1_dates_aligned = pd.to_datetime(self.align_df1.index).normalize()
        df2_dates_aligned = pd.to_datetime(self.align_df2.index).normalize()
        iv_dates_aligned = pd.to_datetime(self.clean_data['Exchange Date']).dt.normalize()
        smile_dates_aligned = pd.to_datetime(self.smile_df.index).normalize()

        common_dates = df1_dates_aligned.intersection(iv_dates_aligned).intersection(df2_dates_aligned).intersection(smile_dates_aligned)

        df1 = self.align_df1[pd.to_datetime(self.align_df1.index).normalize().isin(common_dates)]
        df2 = self.align_df2[pd.to_datetime(self.align_df2.index).normalize().isin(common_dates)]
        smile_df = self.smile_df[pd.to_datetime(self.smile_df.index).normalize().isin(common_dates)]
        self.clean_data = self.clean_data[self.clean_data['Exchange Date'].dt.normalize().isin(common_dates)]

        return df1, df2, self.clean_data, smile_df 

    def get_clean_data(self):

        self.clean_data = self.data.copy()
        self.clean_data = self.clean_data[::-1]
        self.clean_data['Exchange Date'] = pd.to_datetime(self.clean_data['Exchange Date'])
        self.clean_data['Bid'] = pd.to_numeric(self.clean_data['Bid'], errors='coerce')
        self.clean_data['Ask'] = pd.to_numeric(self.clean_data['Ask'], errors='coerce')
        self.clean_data.dropna(inplace=True)

        mask = (self.clean_data['Exchange Date'] >= self.start_date) & (self.clean_data['Exchange Date'] <= self.end_date)
        self.clean_data = self.clean_data[mask]

        self.clean_data['Mid'] = (self.clean_data['Bid'] + self.clean_data['Ask']) / 2
        self.clean_data = self.clean_data.drop(columns=['BidNet'])

        self.clean_data = self.remove_holidays()

        df1, df2, _, smile_df = self.align_datasets()

        return self.clean_data, df1, df2, smile_df

def get_x_mo_realised_vol(realised_variance, window=30):

    N = len(realised_variance)
    valid_window = N - window

    realised_vols_x_mo = np.zeros(valid_window)
    vals = realised_variance.values

    for i in range(valid_window):
        realised_vols_x_mo[i] = np.sqrt(np.sum(vals[i:i+window]))

    dates = realised_variance.index[window:]
    realised_vols_x_mo = pd.Series(realised_vols_x_mo, index=dates)

    annualiser = np.sqrt(252 / window)
    realised_vols_x_mo = realised_vols_x_mo * annualiser * 100

    return realised_vols_x_mo

class Simple_Trading_Strategy_long_short:

    def __init__(self, 
                forecasts: np.ndarray, 
                test_implied_vol: np.ndarray, 
                initial_capital: float, 
                percent_risked: float, 
                r_base: np.ndarray, 
                r_term: np.ndarray, 
                spot_price: np.ndarray, 
                length_of_option: float, 
                long_thresh: float, 
                short_thresh: float,
                overnight_domestic_rate: np.ndarray, 
                margin_percent: float, 
                signal_strength: np.ndarray, 
                signal_multiplier: float, 
                signal_strength_lb: float, 
                signal_strength_up: float
                ) -> None:
        
        self.forecasts = forecasts
        self.test_implied_vol = test_implied_vol
        self.initial_capital = initial_capital
        self.percent_risked = percent_risked
        self.r_base = r_base
        self.r_term = r_term
        self.spot_price = spot_price
        self.length_of_option = length_of_option
        self.long_thresh = long_thresh
        self.short_thresh = short_thresh
        self.overnight_domestic_rate = overnight_domestic_rate
        self.margin_percent = margin_percent
        self.signal_strength = signal_strength
        self.signal_multiplier = signal_multiplier
        self.signal_strength_lb = signal_strength_lb
        self.signal_strength_up = signal_strength_up


    def get_option_value(self, t,  vol, spot, r_b, r_t, K, type='call'):

        eps = 1e-300

        d1 = (np.log(spot / K) + (r_t - r_b + 0.5 * vol**2) * (self.length_of_option - t)) \
                    / (vol * np.sqrt(self.length_of_option - t) + eps)
        
        d2 = d1 - vol * np.sqrt(self.length_of_option - t)

        if type == 'call':
            theta = 1
        elif type == 'put':
            theta = -1
        else:
            raise ValueError("Option type must be 'call' or 'put'.")
        
        option_value = theta * (spot * np.exp(-r_b * (self.length_of_option - t)) * 
                        norm.cdf(theta * d1) - K * np.exp(-r_t * (self.length_of_option - t)) * 
                        norm.cdf(theta * d2))
        
        return option_value
    
    def get_straddle_value(self, t, vol, spot, r_b, r_t, K):
        '''
        Here t is time in years since we brought the option ie what the option is worth at time t. 
        '''

        call_value = self.get_option_value(t, vol, spot, r_b, r_t, K, type='call')
        put_value = self.get_option_value(t, vol, spot, r_b, r_t, K, type='put')

        straddle_value = call_value + put_value

        return straddle_value
    
    def enter_straddle(self, capital, tilde_vol_t, spot_t, r_b_t, r_t_t, K, remain_option_duration, signal_strength):
        '''
        Computes the total premium to enter a long straddle position at t, how large the entry position size is given percent risked 
        and current capital value, as well as the total number of contracts.

        To get nominal value of the straddle we do contract size * straddle value at t.
        '''
        
        long_straddle_value = self.get_straddle_value(self.length_of_option - remain_option_duration,
                                                  tilde_vol_t, spot_t, r_b_t, r_t_t, K)

        signal_strength_bounded = np.clip(signal_strength, self.signal_strength_lb, self.signal_strength_up)  
        entry_long_position_size = self.percent_risked * capital * signal_strength_bounded
        number_contracts = entry_long_position_size // long_straddle_value

        enter_trade_size = number_contracts * long_straddle_value  # total cash value of the trade

        return enter_trade_size, number_contracts, long_straddle_value

    def exit_straddle(self, number_contracts, tilde_vol_t, spot_t, r_b_t, r_t_t, K, remain_option_duration):
        '''
        Signal strength here needs to be same one as we used to open the trade - it is already acconted for 
        in number of contracts.
        '''

        long_straddle_value = self.get_straddle_value(self.length_of_option - remain_option_duration,
                                                  tilde_vol_t, spot_t, r_b_t, r_t_t, K)
    
        exit_trade_size = long_straddle_value * number_contracts

        return exit_trade_size, long_straddle_value


    def accumulator_mm(self, capital_left, overnight_rate_term):
        '''
        Accumulate capital not used for trading in money market account which earns at the overnight term rate
        ie the domestic rate which for USDSGD is SGD. Lets cash acrue overnight. 
        '''

        res = capital_left * (1 + overnight_rate_term) 

        return res
    
    def compute_portfolio_value_with_no_trades(self):

        V_no_trades = np.zeros(len(self.forecasts) + 1)  
        V_no_trades[0] = self.initial_capital

        for t in range(1, len(self.forecasts) + 1):
            overnight_dom_r_t = self.overnight_domestic_rate.iloc[t-1]
            V_no_trades[t] = V_no_trades[t-1] * (1 + overnight_dom_r_t)  

        return V_no_trades
    
    def run_strategy(self):

        N = len(self.forecasts)
        V = np.zeros(N+1) # list of total portfolio value over time
        V[0] = self.initial_capital # initial total portfolio value

        MM_V = np.zeros(N+1) # list of value of the money market account over time (ie cash not used for trading)
        MM_V[0] = V[0] # all money initially is cash in the money market account

        T_V = np.zeros(N+1) # list of total value of the trading portfolio over time
        T_V[0] = 0 # initial trading portfolio value is zero

        T_PnL = np.zeros(N+1) # list of total trading PnL over time
        T_PnL[0] = 0 # initial trading PnL is zero

        list_num_contracts = [] # list for all number of contract sizes we enter in order - use last entry for exit/expiry/mtm values
        list_enter_trade_size = [] # list of all trade sizes we enter in order - use last entry for exit/expiry/mtm values
        Ks = [] # list of strike prices we enter in order - use last entry for exit/expiry/mtm values
        margins = [] # list of margin requirements for each trade
        list_positions = [] # list of positions we enter in order - use last entry for exit/expiry/mtm values

        position = 0  # 0 = no position, 1 = long straddle, -1 = short straddle
        remain_option_duration = self.length_of_option

        list_positions.append(position)  

        for t in range(1, N+1):
            tilde_vol_t = self.test_implied_vol[t-1]
            spot_t = self.spot_price[t-1]
            r_b_t = self.r_base.iloc[t-1]
            r_t_t = self.r_term.iloc[t-1]
            forecast_t = self.forecasts[t-1] # forecasted vol for t to t+h to compare to true implied at t
            overnight_dom_r_t = self.overnight_domestic_rate.iloc[t-1]
            signal_strength_t = self.signal_strength[t-1] * self.signal_multiplier  

            long_condition = forecast_t >= self.long_thresh * tilde_vol_t
            short_condition = forecast_t <= self.short_thresh * tilde_vol_t
            if position == 1 or position == -1:
                remain_option_duration -= 1 / 360

            if position == 0 and long_condition and t != N: # enter long

                position = 1
                remain_option_duration = self.length_of_option  # reset option duration

                # buy 1 mo ATM option at t=0
                K = spot_t  
                Ks.append(K)
                
                enter_total_trade_size, num_contracts_enter, enter_straddle_value = self.enter_straddle(
                    capital=MM_V[t-1],
                    tilde_vol_t=tilde_vol_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    K=K,
                    remain_option_duration=remain_option_duration ,
                    signal_strength=signal_strength_t)

                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) - enter_total_trade_size
                T_V[t] = enter_total_trade_size 
                V[t] = MM_V[t] + T_V[t]
                list_num_contracts.append(num_contracts_enter)
                list_enter_trade_size.append(enter_total_trade_size)
                T_PnL[t] = T_PnL[t-1]

                print(f"Time {t}: Long straddle initiated with trade size {enter_total_trade_size:.2f}, number of contracts {num_contracts_enter}. New total cash available {MM_V[t]:.2f}")

            elif position == 0 and short_condition and t != N: # enter short

                position = -1
                remain_option_duration = self.length_of_option  # reset option duration

                # buy 1 mo ATM option at t=0
                K = spot_t  
                Ks.append(K)
                
                enter_total_trade_size, num_contracts_enter, enter_straddle_value = self.enter_straddle(
                    capital=MM_V[t-1],
                    tilde_vol_t=tilde_vol_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    K=K,
                    remain_option_duration=remain_option_duration, 
                    signal_strength=signal_strength_t)

                margin_requirement = enter_total_trade_size * self.margin_percent
                margins.append(margin_requirement)
                
                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) - margin_requirement
                T_V[t] = -enter_total_trade_size
                V[t] = MM_V[t] + T_V[t] + enter_total_trade_size   # must include the sellings of straddle to total portolio value even if its locked up
                list_num_contracts.append(num_contracts_enter)     
                list_enter_trade_size.append(enter_total_trade_size)
                T_PnL[t] = T_PnL[t-1]

                print(f"Time {t}: Short straddle initiated with trade size {enter_total_trade_size:.2f}, number of contracts {num_contracts_enter}. New total cash available {MM_V[t]:.2f}")

            elif position == 1 and not long_condition:  # exit
                position = 0

                exit_trade_size, exit_straddle_value = self.exit_straddle(
                    number_contracts=list_num_contracts[-1],
                    tilde_vol_t=tilde_vol_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    K=Ks[-1],
                    remain_option_duration=remain_option_duration)
                
                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) + exit_trade_size
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1] + (exit_trade_size - list_enter_trade_size[-1])

                print(f"Time {t}: Long straddle exited with trade size {exit_trade_size:.6f}, number of contracts {list_num_contracts[-1]}. New total cash available {MM_V[t]:.6f}")

                last_enter_trade_size = list_enter_trade_size[-1]
                print(f"PnL for this trade: {exit_trade_size - last_enter_trade_size:.6f}")

            elif position == -1 and not short_condition:  # exit
                position = 0

                exit_trade_size, exit_straddle_value = self.exit_straddle(
                    number_contracts=list_num_contracts[-1],
                    tilde_vol_t=tilde_vol_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    K=Ks[-1],
                    remain_option_duration=remain_option_duration)

                last_enter_trade_size = list_enter_trade_size[-1]
                last_margin_requirement = margins[-1]
                profit = last_enter_trade_size - exit_trade_size

                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) + last_margin_requirement + profit
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1] + profit

                print(f"Time {t}: Short straddle exited with trade size {exit_trade_size:.6f}, number of contracts {list_num_contracts[-1]}. New total cash available {MM_V[t]:.6f}")

                print(f"PnL for this trade: {-(exit_trade_size - last_enter_trade_size):.6f}")

            elif position == 1 and remain_option_duration <= 0:
                position = 0

                strike = Ks[-1]

                market_call_value_T = np.maximum(spot_t - strike, 0) 
                market_put_value_T = np.maximum(strike - spot_t, 0)

                last_num_contracts = list_num_contracts[-1]
                exit_trade_size = last_num_contracts * (market_call_value_T + market_put_value_T) 

                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) + exit_trade_size
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1] + (exit_trade_size - list_enter_trade_size[-1])

                print(f"Time {t}: Long straddle expired with trade size {exit_trade_size:.6f}, number of contracts {last_num_contracts}. New total cash available {MM_V[t]:.6f}")

                last_enter_trade_size = list_enter_trade_size[-1]
                print(f"PnL for this trade: {exit_trade_size - last_enter_trade_size:.6f}")

            elif position == -1 and remain_option_duration <= 0:
                position = 0

                strike = Ks[-1]

                market_call_value_T = np.maximum(spot_t - strike, 0) 
                market_put_value_T = np.maximum(strike - spot_t, 0)

                last_num_contracts = list_num_contracts[-1]
                exit_trade_size = last_num_contracts * (market_call_value_T + market_put_value_T) 

                last_enter_trade_size = list_enter_trade_size[-1]
                last_margin_requirement = margins[-1]
                profit = last_enter_trade_size - exit_trade_size

                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) + last_margin_requirement + profit
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1] + profit

                print(f"Time {t}: Short straddle expired with trade size {exit_trade_size:.6f}, number of contracts {last_num_contracts}. New total cash available {MM_V[t]:.6f}")

                print(f"PnL for this trade: {-(exit_trade_size - last_enter_trade_size):.6f}")
            
            elif position == 1 and remain_option_duration > 0 and t != N:
                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t)

                last_num_contracts = list_num_contracts[-1]
                mtm_straddle_value = self.get_straddle_value(self.length_of_option - remain_option_duration,
                                                          tilde_vol_t, spot_t, r_b_t, r_t_t, K)

                T_V[t] = (last_num_contracts * mtm_straddle_value)  # total size of trading portfolio given mtm value
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1]  # no PnL change since we are still in the trade
            
            elif position == -1 and remain_option_duration > 0 and t != N:
                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t)

                last_num_contracts = list_num_contracts[-1]
                mtm_straddle_value = self.get_straddle_value(self.length_of_option - remain_option_duration,
                                                          tilde_vol_t, spot_t, r_b_t, r_t_t, K)

                last_enter_trade_size = list_enter_trade_size[-1]
                T_V[t] = -(last_num_contracts * mtm_straddle_value)  # total size of trading portfolio given mtm value
                V[t] = MM_V[t] + T_V[t] + last_enter_trade_size # rememeber to add premium received from selling straddle even if its loocked up
                T_PnL[t] = T_PnL[t-1]  # no PnL change since we are still in the trade

            elif position == 0: # no position is open 
                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t)
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1]

            elif position == 1 and t == N: # position open at the end of the trading period

                position = 0

                exit_trade_size, exit_straddle_value = self.exit_straddle(
                    number_contracts=list_num_contracts[-1],
                    tilde_vol_t=tilde_vol_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    K=Ks[-1],
                    remain_option_duration=remain_option_duration)
                
                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) + exit_trade_size
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1] + (exit_trade_size - list_enter_trade_size[-1])

                print(f"Time {t} (end of session): Long straddle exited with trade size {exit_trade_size:.6f}, number of contracts {list_num_contracts[-1]}. New total cash available {MM_V[t]:.6f}")

                last_enter_trade_size = list_enter_trade_size[-1]
                print(f"PnL for this trade: {exit_trade_size - last_enter_trade_size:.6f}")

            elif position == -1 and t == N: # position open at the end of the trading period

                position = 0

                exit_trade_size, exit_straddle_value = self.exit_straddle(
                    number_contracts=list_num_contracts[-1],
                    tilde_vol_t=tilde_vol_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    K=Ks[-1],
                    remain_option_duration=remain_option_duration)
                
                last_enter_trade_size = list_enter_trade_size[-1]
                last_margin_requirement = margins[-1]
                profit = last_enter_trade_size - exit_trade_size

                MM_V[t] = self.accumulator_mm(MM_V[t-1], overnight_dom_r_t) + profit + last_margin_requirement
                T_V[t] = 0.0
                V[t] = MM_V[t] + T_V[t]
                T_PnL[t] = T_PnL[t-1] + profit

                print(f"Time {t} (end of session): Short straddle exited with trade size {exit_trade_size:.6f}, number of contracts {list_num_contracts[-1]}. New total cash available {MM_V[t]:.6f}")

                print(f"PnL for this trade: {-(exit_trade_size - last_enter_trade_size):.6f}")
            
            list_positions.append(position) 
            #print(MM_V[t], T_V[t], V[t], T_PnL[t])

        print(f"Ensure no long trades active at the end of the trading period: {position == 0}")
        print(f"Final cash available after all trades: {MM_V[-1]:.6f}")

        return MM_V, T_V, V, T_PnL, list_positions
    
    def performance_summary(self):

        MM_V, T_V, V, T_PnL, _ = self.run_strategy()

        print("\n")
        print("\n")

        print("Performance Summary:")
        # get portfolio returns
        returns_V = (V[1:] - V[:-1]) / V[:-1]
        excess_returns = returns_V - self.overnight_domestic_rate.values
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        sharpe_annualised = sharpe * np.sqrt(252)  
        print(f"Sharpe Ratio (Annualised): {sharpe_annualised:.6f}")

        rolling_max = np.maximum.accumulate(V)
        daily_drawdown = V / rolling_max - 1
        max_daily_drawdown = np.minimum.accumulate(daily_drawdown) # returns min value seen so far
        print(f"Max Daily Drawdown: {max_daily_drawdown.min() * 100:.6f}%")

        calmar = np.mean(excess_returns) / -max_daily_drawdown.min() * 252
        print(f"Calmar Ratio: {calmar:.6f}")

        print(f"Daily standard deviation of excess returns (annualised): {np.std(excess_returns) * np.sqrt(252):.6f}")

        # CAGR
        cagr = (V[-1] / V[0]) ** (1 / (len(V) / 252)) - 1
        print(f"CAGR: {cagr*100:.6f}%")

        return sharpe_annualised, max_daily_drawdown.min() * 100, calmar, cagr

def run_strategy_with_params(forecasts, test_implied_vol, r_base, r_term, 
                            overnight_domestic_rate, spot_price, long_thresh, short_thresh,
                            initial_capital=100_000, percent_risked=0.05, signal_multiplier=1, plot=False, model_name='BMSM Backtest'):

    strategy = Simple_Trading_Strategy_long_short(
        forecasts=forecasts,
        test_implied_vol=test_implied_vol,
        initial_capital=initial_capital,
        percent_risked=percent_risked,
        r_base=r_base,
        r_term=r_term,
        spot_price=spot_price,
        length_of_option=1/12,
        long_thresh=long_thresh,
        short_thresh=short_thresh,
        overnight_domestic_rate=overnight_domestic_rate,
        margin_percent=0.3,
        signal_strength=np.ones(len(forecasts)),
        signal_multiplier=signal_multiplier,
        signal_strength_lb=0.5,
        signal_strength_up=2.0
    )

    cash, trading_portfolio_value, total_portfolio_value, cum_trading_pnl, positions = strategy.run_strategy()
    cash_no_trades = strategy.compute_portfolio_value_with_no_trades()
    sharpe, max_dd, calmar, cagr = strategy.performance_summary()

    if plot:
        plt.style.use('seaborn-v0_8-dark')
        plt.figure(figsize=(14, 6))
        plt.plot(cash, label='Cash with Trading', color='blue')
        plt.plot(cash_no_trades, label='Cash with no Trading', color='red', linestyle='--')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Cash Value", fontsize=16)
        plt.title(f"Total Realised Cash over Trading Period (Money Market Account) - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()


        plt.figure(figsize=(14, 6))
        plt.plot(cum_trading_pnl, label='Trading PnL', color='blue')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Trading PnL", fontsize=16)
        plt.title(f"Cumulative Trading PnL over Trading Period - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()
        plt.figure(figsize=(14, 6))

        plt.plot(total_portfolio_value, label='Total Portfolio Value', color='green')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Total Portfolio Value", fontsize=16)
        plt.title(f"Total Mark to Market Portfolio Value Over Time - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()
        plt.figure(figsize=(14, 6))

        plt.plot(trading_portfolio_value, label='Trading Portfolio Value', color='orange')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Trading Portfolio Value", fontsize=16)
        plt.title(f"Total Mark to Market Trading Portfolio Value Over Time - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()
        

    return cash, trading_portfolio_value, total_portfolio_value, cum_trading_pnl, positions, cash_no_trades, sharpe, max_dd, calmar, cagr