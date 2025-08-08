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



class Hedging_Trading_Strategy:
    '''
    One option/underlying contract is 100_000 units.
    '''

    def __init__(self, 
                forecasts: np.ndarray, 
                test_implied_vol: np.ndarray, 
                test_vol_smile: pd.DataFrame,
                initial_capital: float, 
                r_base: np.ndarray, 
                r_term: np.ndarray, 
                spot_price: np.ndarray, 
                length_of_option: float, 
                long_thresh: float, 
                short_thresh: float,
                overnight_domestic_rate: np.ndarray, 
                signal_strength: np.ndarray, 
                signal_multiplier: float, 
                signal_strength_lb: float, 
                signal_strength_up: float, 
                base_notional_option: int    # max amount to trade
                ) -> None:
        
        self.forecasts = forecasts
        self.test_implied_vol = test_implied_vol
        self.initial_capital = initial_capital
        self.r_base = r_base
        self.r_term = r_term
        self.spot_price = spot_price
        self.length_of_option = length_of_option
        self.long_thresh = long_thresh
        self.short_thresh = short_thresh
        self.overnight_domestic_rate = overnight_domestic_rate
        self.signal_strength = signal_strength
        self.signal_multiplier = signal_multiplier
        self.signal_strength_lb = signal_strength_lb
        self.signal_strength_up = signal_strength_up
        self.base_notional_option = base_notional_option
        self.test_vol_smile = test_vol_smile

        self.trading_summary = {
            'Time': [],
            'Notional Adjusted': [],
            'Enter Option Trade Size': [],
            'Strike Prices': [],
            'Option Positions': [], 
            'Total Number of Trades': 0, 
            'Enter Hedge Price': []}
        
        N = len(self.forecasts)

        self.option_delta_position = np.zeros(N + 1)  # total delta position of the option at time t (scaled by notional adjusted)
        self.option_delta_position[0] = 0.0

        self.total_hedge_position = np.zeros(N + 1)  # total delta hedge position at time t
        self.total_hedge_position[0] = 0.0

        self.total_adjustments = np.zeros(N + 1)  # total adjustments to delta hedge at time t
        self.total_adjustments[0] = 0.0

        self.adjustment_cashflow = np.zeros(N + 1)  # cash flow from adjustments to delta hedge at time t
        self.adjustment_cashflow[0] = 0.0

        self.MM_V = np.zeros(N + 1)  # money market account value at time t
        self.MM_V[0] = self.initial_capital

        self.V = np.zeros(N + 1)  # total portfolio value at time t
        self.V[0] = self.initial_capital

        self.trading_V = np.zeros(N + 1)  # trading portfolio value at time t (mtm)
        self.trading_V[0] = 0.0

        self.hedge_carry = np.zeros(N + 1)  # hedge carry at time t
        self.hedge_carry[0] = 0.0

        self.daily_option_pnl_mtm = np.zeros(N + 1)
        self.daily_option_pnl_mtm[0] = 0.0

        self.daily_hedge_pnl_mtm = np.zeros(N + 1)
        self.daily_hedge_pnl_mtm[0] = 0.0

        self.daily_option_pnl_real = np.zeros(N + 1)
        self.daily_option_pnl_real[0] = 0.0

        self.option_mtm_value = np.zeros(N + 1) # keep track of mtm for mtm daily profit on options
        self.option_mtm_value[0] = 0.0

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
    
    def enter_straddle(self, tilde_vol_t, spot_t, r_b_t, r_t_t, K, remain_option_duration, signal_strength):
        '''
        Computes the total premium to enter a long straddle position at t, how large the entry position size is given adjusted notional
        value determined from signal strength.
        '''
        
        unit_straddle_value = self.get_straddle_value(self.length_of_option - remain_option_duration,
                                                  tilde_vol_t, spot_t, r_b_t, r_t_t, K) 

        signal_strength_bounded = np.clip(signal_strength, self.signal_strength_lb, self.signal_strength_up)  
        notional_adjusted = self.base_notional_option * signal_strength_bounded

        enter_trade_size = notional_adjusted * unit_straddle_value  # total cash value of the trade

        return enter_trade_size, notional_adjusted, unit_straddle_value

    def exit_straddle(self, notional_adjusted, tilde_vol_t, spot_t, r_b_t, r_t_t, K, remain_option_duration):
        '''
        Signal strength here needs to be same one as we used to open the trade.
        '''

        unit_straddle_value = self.get_straddle_value(self.length_of_option - remain_option_duration,
                                                  tilde_vol_t, spot_t, r_b_t, r_t_t, K)
        
        exit_trade_size = notional_adjusted * unit_straddle_value

        return exit_trade_size, unit_straddle_value


    def accumulator_mm(self, idx, capital_left, overnight_rate_term):
        '''
        Accumulate capital not used for trading in money market account which earns at the overnight term rate (domestic)
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
    
    def get_delta_of_european(self, t, vol, spot, r_b, r_t, K, type='call'):
        '''
        Compute delta of a single call or put option at time t.
        '''

        eps = 1e-300

        d1 = (np.log(spot / K) + (r_t - r_b + 0.5 * vol**2) * (self.length_of_option - t)) \
                    / (vol * np.sqrt(self.length_of_option - t) + eps)
        
        if type == 'call':
            delta = np.exp(-r_b * (self.length_of_option - t)) * norm.cdf(d1)
        elif type == 'put':
            delta = np.exp(-r_b * (self.length_of_option - t)) * (norm.cdf(d1) - 1)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")
        
        return delta
    
    def get_delta_of_straddle(self, t, vol, spot, r_b, r_t, K):
        '''
        Compute delta of a straddle at time t by summing individual deltas - ie this is long short is negative.
        '''

        call_delta = self.get_delta_of_european(t, vol, spot, r_b, r_t, K, type='call')
        put_delta = self.get_delta_of_european(t, vol, spot, r_b, r_t, K, type='put')

        unit_straddle_delta = call_delta + put_delta

        return unit_straddle_delta
    

    def get_delta_hedge_size(self, t, vol, spot, r_b, r_t, K, notional_adjusted):
        '''
        Computes the size of the delta hedge at time t given the notional. Returns delta size to match notional of the option
        given we are long the straddle - minus for short. 
        '''

        unit_straddle_delta = self.get_delta_of_straddle(t, vol, spot, r_b, r_t, K)

        delta_hedge_size = notional_adjusted * unit_straddle_delta

        return delta_hedge_size
    
    def open_long_position(self, idx, spot_t, overnight_dom_r_t, tilde_vol_t, r_b_t, r_t_t, signal_strength_t):
        '''
        This function does all the calculations needed to open a long position in a straddle. We update position, option duration, and all portfolio
        values accordingly. Keep track of trade details for closure and hedging. 
        '''

        self.trading_summary['Time'].append(idx)

        position = 1
        self.trading_summary['Option Positions'].append(position)
        remain_option_duration = self.length_of_option  # reset option duration

        K = spot_t
        self.trading_summary['Strike Prices'].append(K)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], overnight_dom_r_t)

        total_trade_size, notional_adjusted, unit_straddle_value = self.enter_straddle(
            tilde_vol_t = tilde_vol_t,
            spot_t = spot_t,
            r_b_t = r_b_t,
            r_t_t = r_t_t,
            K = K,
            remain_option_duration = remain_option_duration,
            signal_strength = signal_strength_t)

        self.trading_summary['Notional Adjusted'].append(notional_adjusted)
        self.trading_summary['Enter Option Trade Size'].append(total_trade_size)

        self.hedge_carry[idx] = 0.0  # no carry at start
        self.option_delta_position[idx] = 0.0 # delta neutral at start for ATM straddle
        self.total_hedge_position[idx] = 0.0  # no hedge position at start
        self.total_adjustments[idx] = 0.0 
        self.adjustment_cashflow[idx] = 0.0

        self.MM_V[idx] -= total_trade_size  # deduct cost of trade from cash
        self.trading_V[idx] = total_trade_size  
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]  

        self.daily_hedge_pnl_mtm[idx] = 0.0
        self.option_mtm_value[idx] = total_trade_size
        self.daily_option_pnl_mtm[idx] = 0.0
        self.daily_option_pnl_real[idx] = 0.0 

        print(f"Time {idx}: Long straddle initiated with trade size {total_trade_size:.2f}, notional adjusted {notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        self.trading_summary['Total Number of Trades'] += 1  

        return position, remain_option_duration

    def open_short_position(self, idx, spot_t, overnight_dom_r_t, tilde_vol_t, r_b_t, r_t_t, signal_strength_t):
        '''
        Same as above but for short straddle.
        '''

        self.trading_summary['Time'].append(idx)

        position = -1
        self.trading_summary['Option Positions'].append(position)
        remain_option_duration = self.length_of_option  # reset option duration

        K = spot_t
        self.trading_summary['Strike Prices'].append(K)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], overnight_dom_r_t)

        total_trade_size, notional_adjusted, unit_straddle_value = self.enter_straddle(
            tilde_vol_t = tilde_vol_t,
            spot_t = spot_t,
            r_b_t = r_b_t,
            r_t_t = r_t_t,
            K = K,
            remain_option_duration = remain_option_duration,
            signal_strength = signal_strength_t)

        self.trading_summary['Notional Adjusted'].append(notional_adjusted)
        self.trading_summary['Enter Option Trade Size'].append(total_trade_size)

        self.hedge_carry[idx] = 0.0
        self.option_delta_position[idx] = 0.0 # delta neutral at start for ATM straddle
        self.total_hedge_position[idx] = 0.0
        self.total_adjustments[idx] = 0.0
        self.adjustment_cashflow[idx] = 0.0

        self.MM_V[idx] += total_trade_size  # add cash from entering short position
        self.trading_V[idx] = -total_trade_size  # negative value for short position
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        self.daily_hedge_pnl_mtm[idx] = 0.0
        self.option_mtm_value[idx] = -total_trade_size
        self.daily_option_pnl_mtm[idx] = 0.0
        self.daily_option_pnl_real[idx] = 0.0

        print(f"Time {idx}: Short straddle initiated with trade size {total_trade_size:.2f}, notional adjusted {notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}") 
        
        self.trading_summary['Total Number of Trades'] += 1  

        return position, remain_option_duration
    
    def get_vol_smile_strikes(self, vol_smile_t, spot_t, r_b_t, r_t_t):
        '''
        Get the vol smile in terms of strikes so we can compare to the option strikes we have entered at. 
        '''

        deltas = pd.to_numeric(vol_smile_t.index, errors='coerce')
        strikes = np.zeros_like(deltas)
        tau = self.length_of_option

        for i, delta in enumerate(deltas):

            vol = vol_smile_t.iloc[i]

            if np.isnan(delta):
                inv_term = np.exp(r_b_t * tau) * -0.5 + 1
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strike_put = spot_t / np.exp(exp_term)

                inv_term = np.exp(r_b_t * tau) * 0.5
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strike_call = spot_t / np.exp(exp_term)

                strikes[i] = (strike_put + strike_call) / 2  


            elif delta < 0:
                inv_term = np.exp(r_b_t * tau) * delta + 1
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strikes[i] = spot_t / np.exp(exp_term)
            
            elif delta > 0:
                inv_term = np.exp(r_b_t * tau) * delta
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strikes[i] = spot_t / np.exp(exp_term)

        vols = vol_smile_t.values
        strike_smile = pd.Series(vols, index=strikes)

        return strike_smile
    
    def get_interpolated_vol(self, vol_smile_strikes_t, k_bar):
        '''
        Get interpolated value from cubic spline and interpolate linearly if outside range. 
        '''

        spline = CubicSpline(x = vol_smile_strikes_t.index, y = vol_smile_strikes_t.values, bc_type='natural', extrapolate=False)
        result = spline(k_bar)

        if not np.isnan(result):

            return result
        else:
            linear = interp1d(x = vol_smile_strikes_t.index, y = vol_smile_strikes_t.values, kind = 'linear', bounds_error=False, fill_value='extrapolate')
            linear_result = linear(k_bar)

            return linear_result
        
    def exit_long_position_signal(self, idx, vol_smile_t, spot_t, r_b_t, r_t_t, overnight_dom_r_t, remain_option_duration):
        '''
        Exit a long straddle position given the signal is no longer valid.
        '''

        position = 0
        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(position)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], overnight_dom_r_t)

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_enter_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]

        # get the correct implied vols from the smile
        vol_smile_strikes = self.get_vol_smile_strikes(
                                                    vol_smile_t=vol_smile_t,
                                                    spot_t=spot_t,
                                                    r_b_t=r_b_t,
                                                    r_t_t=r_t_t)
        
        tilde_vol_from_smile = self.get_interpolated_vol(
            vol_smile_strikes_t=vol_smile_strikes,
            k_bar=last_strike)

        exit_trade_size, exit_straddle_value = self.exit_straddle(
            notional_adjusted=last_notional_adjusted,
            tilde_vol_t=tilde_vol_from_smile,
            spot_t=spot_t,
            r_b_t=r_b_t,
            r_t_t=r_t_t,
            K=last_strike,
            remain_option_duration=remain_option_duration)
        
        # exit the hedge position as well
        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += exit_trade_size + self.adjustment_cashflow[idx]  
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        print(f"Time {idx}: Long straddle exited with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        option_pnl = exit_trade_size - last_enter_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]

        print(f"Option PnL for this trade: {option_pnl:.2f}")

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 

        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position
    
    def exit_short_position_signal(self, idx, vol_smile_t, spot_t, r_b_t, r_t_t, overnight_dom_r_t, remain_option_duration):
        '''
        Exit a short straddle position given the signal is no longer valid.
        '''

        position = 0
        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(position)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], overnight_dom_r_t)

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_enter_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]

        #get the correct implied vols from the smile
        vol_smile_strikes = self.get_vol_smile_strikes(
                                                    vol_smile_t=vol_smile_t,
                                                    spot_t=spot_t,
                                                    r_b_t=r_b_t,
                                                    r_t_t=r_t_t)
        
        tilde_vol_from_smile = self.get_interpolated_vol(
            vol_smile_strikes_t=vol_smile_strikes,
            k_bar=last_strike)

        exit_trade_size, exit_straddle_value = self.exit_straddle(
            notional_adjusted=last_notional_adjusted,
            tilde_vol_t=tilde_vol_from_smile,
            spot_t=spot_t,
            r_b_t=r_b_t,
            r_t_t=r_t_t,
            K=last_strike,
            remain_option_duration=remain_option_duration)
        
        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += -exit_trade_size + self.adjustment_cashflow[idx]
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        print(f"Time {idx}: Short straddle exited with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        option_pnl = last_enter_trade_size - exit_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = -exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]
        print(f"Option PnL for this trade: {option_pnl:.2f}")

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 
        
        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position
    

    def exit_long_position_expiry(self, idx, spot_t):
        
        position = 0
        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(position)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_enter_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]

        call_T = np.maximum(spot_t - last_strike, 0)
        put_T = np.maximum(last_strike - spot_t, 0)

        exit_trade_size = last_notional_adjusted * (call_T + put_T)

        # exit the hedge position as well
        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += exit_trade_size + self.adjustment_cashflow[idx]  
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        print(f"Time {idx}: Long straddle expired with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        option_pnl = exit_trade_size - last_enter_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]
        print(f"Option PnL for this trade: {option_pnl:.2f}")

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 
        
        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position

    def exit_short_position_expiry(self, idx, spot_t):

        position = 0

        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(position)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_enter_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]

        call_T = np.maximum(spot_t - last_strike, 0)
        put_T = np.maximum(last_strike - spot_t, 0)

        exit_trade_size = last_notional_adjusted * (call_T + put_T)

        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += -exit_trade_size + self.adjustment_cashflow[idx]
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        print(f"Time {idx}: Short straddle expired with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        option_pnl = last_enter_trade_size - exit_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = -exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]
        print(f"Option PnL for this trade: {option_pnl:.2f}")

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 
        
        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position

    def mark_to_market_long_position(self, idx, vol_smile_t, spot_t, r_b_t, r_t_t, remain_option_duration):
        '''
        Mark to market a long straddle position at time t given the current market conditions - consider delta hedge as well. 
        '''

        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(1)  # still long position

        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])  # carry cash from overnight

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]
        self.trading_summary['Strike Prices'].append(last_strike)  # keep the last strike price

        last_entry_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        self.trading_summary['Enter Option Trade Size'].append(last_entry_trade_size)  
        self.trading_summary['Notional Adjusted'].append(last_notional_adjusted) 

        #get the correct implied vols from the smile
        vol_smile_strikes = self.get_vol_smile_strikes(
                                                    vol_smile_t=vol_smile_t,
                                                    spot_t=spot_t,
                                                    r_b_t=r_b_t,
                                                    r_t_t=r_t_t)
        
        tilde_vol_from_smile = self.get_interpolated_vol(
            vol_smile_strikes_t=vol_smile_strikes,
            k_bar=last_strike)

        mtm_unit_straddle_value = self.get_straddle_value(
            t=self.length_of_option - remain_option_duration,
            vol=tilde_vol_from_smile,
            spot=spot_t,
            r_b=r_b_t,
            r_t=r_t_t,
            K=last_strike)

        mtm_trade_size = last_notional_adjusted * mtm_unit_straddle_value

        delta_size = self.get_delta_hedge_size(
            t=self.length_of_option - remain_option_duration,
            vol=tilde_vol_from_smile,
            spot=spot_t,
            r_b=r_b_t,
            r_t=r_t_t,
            K=last_strike,
            notional_adjusted=last_notional_adjusted)

        self.option_delta_position[idx] = delta_size #option delta position is pos value of this as long

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx] # hedge is negative of position delta
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx-1])  
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = mtm_trade_size + (self.total_hedge_position[idx] * spot_t)
        self.MM_V[idx] += self.adjustment_cashflow[idx]   
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx] 

        self.daily_option_pnl_real[idx] = 0.0

        self.option_mtm_value[idx] = mtm_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 

    def mark_to_market_short_position(self, idx, vol_smile_t, spot_t, r_b_t, r_t_t, remain_option_duration):
        '''
        Mark to market a long straddle position at time t given the current market conditions - consider delta hedge as well. 
        '''

        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(-1)  # still short position

        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])  # carry cash from overnight

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]
        self.trading_summary['Strike Prices'].append(last_strike)  # keep the last strike price

        last_entry_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        self.trading_summary['Enter Option Trade Size'].append(last_entry_trade_size)  
        self.trading_summary['Notional Adjusted'].append(last_notional_adjusted) 

        #get the correct implied vols from the smile
        vol_smile_strikes = self.get_vol_smile_strikes(
                                                    vol_smile_t=vol_smile_t,
                                                    spot_t=spot_t,
                                                    r_b_t=r_b_t,
                                                    r_t_t=r_t_t)
        
        tilde_vol_from_smile = self.get_interpolated_vol(
            vol_smile_strikes_t=vol_smile_strikes,
            k_bar=last_strike)

        mtm_unit_straddle_value = self.get_straddle_value(
            t=self.length_of_option - remain_option_duration,
            vol=tilde_vol_from_smile,
            spot=spot_t,
            r_b=r_b_t,
            r_t=r_t_t,
            K=last_strike)

        mtm_trade_size = last_notional_adjusted * mtm_unit_straddle_value

        delta_size = self.get_delta_hedge_size(
            t=self.length_of_option - remain_option_duration,
            vol=tilde_vol_from_smile,
            spot=spot_t,
            r_b=r_b_t,
            r_t=r_t_t,
            K=last_strike,
            notional_adjusted=last_notional_adjusted)

        self.option_delta_position[idx] = -delta_size # minus as we are shorting the straddle

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx] # hedge is negative of position delta
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx-1])  
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = -mtm_trade_size + (self.total_hedge_position[idx] * spot_t)
        self.MM_V[idx] += self.adjustment_cashflow[idx] 
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        self.daily_option_pnl_real[idx] = 0.0

        self.option_mtm_value[idx] = -mtm_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 

    def exit_long_position_end_of_session(self, idx, vol_smile_t, spot_t, r_b_t, r_t_t, remain_option_duration):

        position = 0

        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(position)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_enter_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]

        #get the correct implied vols from the smile
        vol_smile_strikes = self.get_vol_smile_strikes(
                                                    vol_smile_t=vol_smile_t,
                                                    spot_t=spot_t,
                                                    r_b_t=r_b_t,
                                                    r_t_t=r_t_t)
        
        tilde_vol_from_smile = self.get_interpolated_vol(
            vol_smile_strikes_t=vol_smile_strikes,
            k_bar=last_strike)

        exit_trade_size, exit_straddle_value = self.exit_straddle(
            notional_adjusted=last_notional_adjusted,
            tilde_vol_t=tilde_vol_from_smile,
            spot_t=spot_t,
            r_b_t=r_b_t,
            r_t_t=r_t_t,
            K = last_strike,
            remain_option_duration=remain_option_duration)
        
        # exit the hedge position as well
        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += exit_trade_size + self.adjustment_cashflow[idx]  
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        print(f"Time {idx}: Long straddle exited (end of session) with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        option_pnl = exit_trade_size - last_enter_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]
        print(f"Option PnL for this trade: {option_pnl:.2f}")

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 
        
        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position
        

    def exit_short_position_end_of_session(self, idx, vol_smile_t, spot_t, r_b_t, r_t_t, remain_option_duration):
        
        position = 0

        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(position)

        # carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])

        last_notional_adjusted = self.trading_summary['Notional Adjusted'][-1]
        last_enter_trade_size = self.trading_summary['Enter Option Trade Size'][-1]
        last_strike = self.trading_summary['Strike Prices'][-1]

        #get the correct implied vols from the smile
        vol_smile_strikes = self.get_vol_smile_strikes(
                                                    vol_smile_t=vol_smile_t,
                                                    spot_t=spot_t,
                                                    r_b_t=r_b_t,
                                                    r_t_t=r_t_t)
        
        tilde_vol_from_smile = self.get_interpolated_vol(
            vol_smile_strikes_t=vol_smile_strikes,
            k_bar=last_strike)

        exit_trade_size, exit_straddle_value = self.exit_straddle(
            notional_adjusted=last_notional_adjusted,
            tilde_vol_t=tilde_vol_from_smile,
            spot_t=spot_t,
            r_b_t=r_b_t,
            r_t_t=r_t_t,
            K = last_strike,
            remain_option_duration=remain_option_duration)
        
        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += -exit_trade_size + self.adjustment_cashflow[idx]
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        print(f"Time {idx}: Short straddle exited (end of session) with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")

        option_pnl = last_enter_trade_size - exit_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = -exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]
        print(f"Option PnL for this trade: {option_pnl:.2f}")

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 
        
        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position
    
    def no_position_open(self, idx):

        self.trading_summary['Time'].append(idx)
        self.trading_summary['Option Positions'].append(0)
        self.trading_summary['Enter Option Trade Size'].append(0.0)  
        self.trading_summary['Strike Prices'].append(np.nan)  
        self.trading_summary['Notional Adjusted'].append(0.0)  

        # no position, just carry cash from overnight
        self.MM_V[idx] = self.accumulator_mm(idx, self.MM_V[idx-1], self.overnight_domestic_rate.iloc[idx-1])

        self.option_delta_position[idx] = 0.0
        self.total_hedge_position[idx] = 0.0
        self.total_adjustments[idx] = 0.0
        self.adjustment_cashflow[idx] = 0.0

        self.daily_hedge_pnl_mtm[idx] = 0.0
        self.daily_option_pnl_real[idx] = 0.0
        self.daily_option_pnl_mtm[idx] = 0.0
        self.option_mtm_value[idx] = 0.0

        self.trading_V[idx] = 0.0
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]
    
    def run_strategy(self):

        N = len(self.forecasts)

        position = 0  # 0 = no position, 1 = long straddle, -1 = short straddle
        remain_option_duration = self.length_of_option

        self.trading_summary['Time'].append(0)
        self.trading_summary['Notional Adjusted'].append(0)
        self.trading_summary['Enter Option Trade Size'].append(0)
        self.trading_summary['Strike Prices'].append(np.nan)
        self.trading_summary['Option Positions'].append(position)

        for t in range(1, N+1):
            tilde_vol_t = self.test_implied_vol[t-1]
            spot_t = self.spot_price[t-1]
            r_b_t = self.r_base.iloc[t-1]
            r_t_t = self.r_term.iloc[t-1]
            forecast_t = self.forecasts[t-1] # forecasted vol for t to t+h to compare to true implied at t
            overnight_dom_r_t = self.overnight_domestic_rate.iloc[t-1]
            signal_strength_t = self.signal_strength[t-1] * self.signal_multiplier  
            vol_smile_t = self.test_vol_smile.iloc[t-1]  

            long_condition = forecast_t >= self.long_thresh * tilde_vol_t
            short_condition = forecast_t <= self.short_thresh * tilde_vol_t
            if position == 1 or position == -1:
                remain_option_duration -= 1 / 360

            if position == 0 and long_condition and t != N: 

                position, remain_option_duration = self.open_long_position(
                    idx=t,
                    spot_t=spot_t,
                    overnight_dom_r_t=overnight_dom_r_t,
                    tilde_vol_t=tilde_vol_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    signal_strength_t=signal_strength_t)
                
            elif position == 0 and short_condition and t != N: 

                position, remain_option_duration = self.open_short_position(
                    idx=t,
                    spot_t=spot_t,
                    overnight_dom_r_t=overnight_dom_r_t,
                    tilde_vol_t=tilde_vol_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    signal_strength_t=signal_strength_t)
                
            elif position == 1 and not long_condition:  

                position = self.exit_long_position_signal(
                    idx=t,
                    vol_smile_t=vol_smile_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    overnight_dom_r_t=overnight_dom_r_t,
                    remain_option_duration=remain_option_duration)


            elif position == -1 and not short_condition: 
                
                position = self.exit_short_position_signal(
                    idx=t,
                    vol_smile_t=vol_smile_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    overnight_dom_r_t=overnight_dom_r_t,
                    remain_option_duration=remain_option_duration)

            elif position == 1 and remain_option_duration <= 0:

                position = self.exit_long_position_expiry(
                    idx=t,
                    spot_t=spot_t)

            elif position == -1 and remain_option_duration <= 0:

                position = self.exit_short_position_expiry(
                    idx=t,
                    spot_t=spot_t)
            
            elif position == 1 and remain_option_duration > 0 and t != N:
                
                self.mark_to_market_long_position(
                    idx=t,
                    vol_smile_t=vol_smile_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    remain_option_duration=remain_option_duration)
            
            elif position == -1 and remain_option_duration > 0 and t != N:
                
                self.mark_to_market_short_position(
                    idx=t,
                    vol_smile_t=vol_smile_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    remain_option_duration=remain_option_duration)

            elif position == 0: 

                self.no_position_open(idx=t)

            elif position == 1 and t == N: # position open at the end of the trading period

                position = self.exit_long_position_end_of_session(
                    idx=t,
                    vol_smile_t=vol_smile_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    remain_option_duration=remain_option_duration)

            elif position == -1 and t == N: # position open at the end of the trading period

                position = self.exit_short_position_end_of_session(
                    idx=t,
                    vol_smile_t=vol_smile_t,
                    spot_t=spot_t,
                    r_b_t=r_b_t,
                    r_t_t=r_t_t,
                    remain_option_duration=remain_option_duration)

        print(f"Ensure no long trades active at the end of the trading period: {position == 0}")
        print(f"Final cash available after all trades: {self.MM_V[-1]:.6f}")

        return self.MM_V, self.trading_V, self.V, self.trading_summary, self.option_delta_position, self.total_hedge_position, self.adjustment_cashflow, self.hedge_carry, self.daily_option_pnl_real, self.daily_option_pnl_mtm, self.daily_hedge_pnl_mtm
    
    def performance_summary(self):

        MM_V, T_V, V, _, _, _, _, _, _, _, _  = self.run_strategy()

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

def run_strategy_with_params(forecasts, test_implied_vol, test_vol_smile, r_base, r_term, 
                            overnight_domestic_rate, spot_price, long_thresh, short_thresh, signal_strength, length_option=1/12,
                            initial_capital=1_000_000 , signal_multiplier=1, signal_lb = 0.01, signal_ub = 0.99, plot=False, model_name='BMSM Backtest', 
                            base_notional=100_000):

    strategy = Hedging_Trading_Strategy(
        forecasts=forecasts,
        test_implied_vol=test_implied_vol,
        test_vol_smile=test_vol_smile,
        initial_capital=initial_capital,
        r_base=r_base,
        r_term=r_term,
        spot_price=spot_price,
        length_of_option=length_option,
        long_thresh=long_thresh,
        short_thresh=short_thresh,
        overnight_domestic_rate=overnight_domestic_rate,
        signal_strength=signal_strength,
        signal_multiplier=signal_multiplier,
        signal_strength_lb=signal_lb,
        signal_strength_up=signal_ub,
        base_notional_option=base_notional
    )

    cash, trading_portfolio_value, total_portfolio_value, trading_summary, option_delta_positions, delta_hedge, delta_hedge_cashflows, hedge_carry, daily_option_pnl_real, daily_option_pnl_mtm, daily_hedge_pnl_mtm = strategy.run_strategy()
    cash_no_trades = strategy.compute_portfolio_value_with_no_trades()
    sharpe, max_dd, calmar, cagr = strategy.performance_summary()

    if plot:
        plt.style.use('seaborn-v0_8-dark')
        plt.figure(figsize=(14, 6))
        plt.plot(cash, label='Cash with Trading', color='blue')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Cash Value", fontsize=16)
        plt.title(f"Total Cash in Domestic Currency over Trading Period - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.plot(daily_option_pnl_mtm.cumsum(), label='Option PnL', color='blue')
        plt.plot(daily_hedge_pnl_mtm.cumsum(), label='Hedge PnL', color='orange')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Trading PnL", fontsize=16)
        plt.title(f"Cumulative Mark to Market Trading PnL - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.plot(daily_option_pnl_mtm.cumsum(), label='Option PnL MtM', color='orange')
        plt.plot(daily_option_pnl_real.cumsum(), label='Option PnL Realised', color='blue')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Trading PnL", fontsize=16)
        plt.title(f"Cumulative Option Trading PnL - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.plot(total_portfolio_value, label='Total Portfolio Value', color='green')
        plt.plot(cash_no_trades, label='Cash with no Trading', color='red', linestyle='--')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Total Portfolio Value", fontsize=16)
        plt.title(f"Total Mark to Market Portfolio Value Over Time - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()


    return cash, trading_portfolio_value, total_portfolio_value, trading_summary, option_delta_positions, delta_hedge, delta_hedge_cashflows, hedge_carry, daily_option_pnl_real, daily_option_pnl_mtm, daily_hedge_pnl_mtm, sharpe, max_dd, calmar, cagr, cash_no_trades
