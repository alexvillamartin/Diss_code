from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import CubicSpline, interp1d
from itertools import product
from tqdm import tqdm
from estimation_forecast_functions import FIGARCH_Forecaster_1_d_0_MLE_delta_stds, GARCH_Forecaster_MLE_1_1_delta_stds, pre_whitening, GARCH_Forecaster_MLE_p_1_q_2_delta_stds, BMSM_Forecaster_MLE_delta_stds
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import optuna

def regression_forecasts_ridge_library(train_forecasts, train_implied_vols, alpha_implied, train_true_vols, backtest_forecasts, backtest_implied_vols):
    '''
    Forecasts are already one lagged by one day so forecast today from [t, t+h] done with ydays data corresponds to the 
    implied vol observed yday. 
    '''

    train_forecasts = np.asarray(train_forecasts)
    train_implied_vols = np.asarray(train_implied_vols)
    train_true_vols = np.asarray(train_true_vols)

    y_log = np.log(train_true_vols)
    x1_log = np.log(train_forecasts)
    x2_log = np.log(train_implied_vols)
    
    X_features = np.column_stack((x1_log, x2_log))
    X = sm.add_constant(X_features)

    scalar = StandardScaler()
    X[:, 1:] = scalar.fit_transform(X[:, 1:])

    alpha_vector = np.array([0.0, 0.1, alpha_implied]) # minimal penalty to forecast and select based on implied penalty

    model = sm.OLS(y_log, X).fit_regularized(alpha=alpha_vector, L1_wt=0.0)

    # get predictions for use in backtest under this alpha
    X_test = sm.add_constant(np.column_stack((np.log(backtest_forecasts), np.log(backtest_implied_vols))))
    X_test[:, 1:] = scalar.transform(X_test[:, 1:])
    predictions = model.predict(X_test)

    # bias correction using non-parametric smear
    resid_log = y_log - model.predict(X)
    smear = float(np.mean(np.exp(resid_log)))
    vol_pred = np.exp(predictions) * smear

    return vol_pred, smear, model 

def regression_forecasts_ridge_train(train_forecasts, train_implied_vols, alpha_implied, train_true_vols, backtest_forecasts, backtest_implied_vols):
    '''
    Forecasts are already one lagged by one day so forecast today from [t, t+h] done with ydays data corresponds to the 
    implied vol observed yday. 

    Analytical version does MSE so divide by n to match library alphas.
    '''

    train_forecasts = np.asarray(train_forecasts)
    train_implied_vols = np.asarray(train_implied_vols)
    train_true_vols = np.asarray(train_true_vols)

    y_log = np.log(train_true_vols)
    x1_log = np.log(train_forecasts)
    x2_log = np.log(train_implied_vols)
    
    X_features = np.column_stack((x1_log, x2_log))
    X = sm.add_constant(X_features)

    scalar = StandardScaler()
    X[:, 1:] = scalar.fit_transform(X[:, 1:])

    n = X.shape[0]
    XtX = (X.T @ X) / n
    Xty = (X.T @ y_log) / n
    R = np.diag([0.0, 0.1, alpha_implied])

    A = XtX + R
    L = np.linalg.cholesky(A)
    beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))

    yhat_log_train = X @ beta
    resid_log = y_log - yhat_log_train
    smear = float(np.mean(np.exp(resid_log)))

    Xt = sm.add_constant(np.column_stack((np.log(backtest_forecasts), np.log(backtest_implied_vols))))
    Xt[:, 1:] = scalar.transform(Xt[:, 1:])

    yhat_log = Xt @ beta
    vol_pred = np.exp(yhat_log) * smear

    return vol_pred, smear, beta, scalar

def regression_forecasts_ridge_test(forecasts, implied_vols, smear, beta, scalar):

     X = sm.add_constant(np.column_stack((np.log(forecasts), np.log(implied_vols))))
     X[:, 1:] = scalar.transform(X[:, 1:])
     yhat_log = X @ beta
     vol_pred = np.exp(yhat_log) * smear

     return vol_pred

def grid_search_params2(backtest_data, param_grid, init_cap, OPTION_MATURITY, SIGNAL_LB, SIGNAL_UB, 
                       NOTIONAL_USD, MAX_DELTA_DIFF, TRANSACTION_COST_BOOL, TRANSACTION_COSTS_SPOT, TRANSACTION_COSTS_OPTION, verbose, convert_USD, OLS):

    forecasts = backtest_data[0]
    test_implied_vol = backtest_data[1]
    r_base = backtest_data[2]
    r_term = backtest_data[3]
    overnight_domestic_rate = backtest_data[4]
    spot_price = backtest_data[5]
    test_vol_smile = backtest_data[6]
    signal_strength = backtest_data[7]
    overnight_foreign_rate = backtest_data[8]

    train_forecasts_ols = backtest_data[9]
    train_implied_vols_ols = backtest_data[10]
    train_true_vols_ols = backtest_data[11]


    param_combinations = list(product(param_grid['long_threshold'],
                                     param_grid['short_threshold'], 
                                     param_grid['signal_multiplier'], 
                                     param_grid['alpha_implied']))
    
    results = []

    for long_thresh, short_thresh, signal_multiplier, alpha_implied in tqdm(param_combinations, desc = "Parameter Combinations"):

        if OLS:
             
            forecasts, _, _, _ = regression_forecasts_ridge_train(train_forecasts_ols, \
                                                                 train_implied_vols_ols, alpha_implied, \
                                                                 train_true_vols_ols, forecasts, \
                                                                 test_implied_vol)
        else:
             forecasts = forecasts

        strategy = Hedging_Trading_Strategy(
            forecasts=forecasts,
            test_implied_vol=test_implied_vol,
            test_vol_smile=test_vol_smile,
            initial_capital=init_cap,
            r_base=r_base,
            r_term=r_term,
            spot_price=spot_price,
            length_of_option= OPTION_MATURITY,
            long_thresh=long_thresh,
            short_thresh=short_thresh,
            overnight_domestic_rate=overnight_domestic_rate,
            overnight_foreign_rate=overnight_foreign_rate,
            signal_strength=signal_strength,
            signal_multiplier=signal_multiplier,
            signal_strength_lb=SIGNAL_LB,
            signal_strength_up=SIGNAL_UB, 
            base_notional_option=NOTIONAL_USD, 
            directional_risk_max=MAX_DELTA_DIFF, 
            transaction_cost_bool=TRANSACTION_COST_BOOL,
            transaction_costs_spot=TRANSACTION_COSTS_SPOT,
            transaction_costs_option=TRANSACTION_COSTS_OPTION, 
            verbose = verbose, 
            convert_USD=convert_USD)
        
        _ = strategy.run_strategy()

        sharpe, max_dd, calmar, cagr = strategy.performance_summary()

        results.append({
            'long_threshold': long_thresh,
            'short_threshold': short_thresh,
            'signal_multiplier': signal_multiplier,
            'alpha_implied': alpha_implied,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'cagr': cagr})
        
    return pd.DataFrame(results)

def grid_search_params(backtest_data, param_grid, init_cap, OPTION_MATURITY, SIGNAL_LB, SIGNAL_UB, 
                       NOTIONAL_USD, MAX_DELTA_DIFF, TRANSACTION_COST_BOOL, TRANSACTION_COSTS_SPOT, TRANSACTION_COSTS_OPTION, 
                       verbose, convert_USD, OLS):

    forecasts = backtest_data[0]
    test_implied_vol = backtest_data[1]
    r_base = backtest_data[2]
    r_term = backtest_data[3]
    overnight_domestic_rate = backtest_data[4]
    spot_price = backtest_data[5]
    test_vol_smile = backtest_data[6]
    signal_strength = backtest_data[7]
    overnight_foreign_rate = backtest_data[8]
    train_forecasts_ols = backtest_data[9]
    train_implied_vols_ols = backtest_data[10]
    train_true_vols_ols = backtest_data[11]

    longs  = tuple(float(x) for x in param_grid['long_threshold'])
    shorts = tuple(float(x) for x in param_grid['short_threshold'])
    sigmul = tuple(int(x)   for x in param_grid['signal_multiplier'])
    alphas = tuple(float(x) for x in param_grid['alpha_implied']) if OLS else (0.0,)

    n_trials = int(param_grid.get('n_trials', 400))
    seed = int(param_grid.get('seed', 12))
    n_jobs = int(param_grid.get('n_jobs', 1))  

    def run_once(long_thresh, short_thresh, signal_multiplier, alpha_implied):

        if OLS:
            fc_use, _, _, _ = regression_forecasts_ridge_train(
                train_forecasts_ols, train_implied_vols_ols, alpha_implied,
                train_true_vols_ols, forecasts, test_implied_vol
            )
        else:
            fc_use = forecasts

        strategy = Hedging_Trading_Strategy(
            forecasts=fc_use,
            test_implied_vol=test_implied_vol,
            test_vol_smile=test_vol_smile,
            initial_capital=init_cap,
            r_base=r_base,
            r_term=r_term,
            spot_price=spot_price,
            length_of_option=OPTION_MATURITY,
            long_thresh=long_thresh,
            short_thresh=short_thresh,
            overnight_domestic_rate=overnight_domestic_rate,
            overnight_foreign_rate=overnight_foreign_rate,
            signal_strength=signal_strength,
            signal_multiplier=signal_multiplier,
            signal_strength_lb=SIGNAL_LB,
            signal_strength_up=SIGNAL_UB,
            base_notional_option=NOTIONAL_USD,
            directional_risk_max=MAX_DELTA_DIFF,
            transaction_cost_bool=TRANSACTION_COST_BOOL,
            transaction_costs_spot=TRANSACTION_COSTS_SPOT,
            transaction_costs_option=TRANSACTION_COSTS_OPTION,
            verbose=False,  
            convert_USD=convert_USD
        )
        _ = strategy.run_strategy()
        sharpe, max_dd, calmar, cagr, precision_all, precision_active, win_loss_ratio_all, win_loss_ratio_active,\
              avg_win_all, avg_win_active, avg_loss_all, avg_loss_active, non_active_share, total_trades = strategy.performance_summary()
        return float(sharpe), float(max_dd), float(calmar), float(cagr)

    def objective(trial: optuna.Trial):

        long_thresh = trial.suggest_categorical("long_threshold", longs)
        short_thresh = trial.suggest_categorical("short_threshold", shorts)
        signal_multiplier = trial.suggest_categorical("signal_multiplier", sigmul)
        alpha_implied = trial.suggest_categorical("alpha_implied", alphas)

        sharpe, max_dd, calmar, cagr = \
                    run_once(long_thresh, short_thresh, signal_multiplier, alpha_implied)

        trial.set_user_attr("max_drawdown", float(max_dd))
        trial.set_user_attr("calmar_ratio", float(calmar))
        trial.set_user_attr("cagr", float(cagr))

        return sharpe 

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        p = t.params
        rows.append({
            'long_threshold': p['long_threshold'],
            'short_threshold': p['short_threshold'],
            'signal_multiplier': p['signal_multiplier'],
            'alpha_implied': p.get('alpha_implied', 0.0),
            'sharpe_ratio': float(t.value),
            'max_drawdown': float(t.user_attrs.get('max_drawdown', np.nan)),
            'calmar_ratio': float(t.user_attrs.get('calmar_ratio', np.nan)),
            'cagr': float(t.user_attrs.get('cagr', np.nan)),
        })

    df = pd.DataFrame(rows)

    return df

def run_strategy_with_params(forecasts, test_implied_vol, test_vol_smile, r_base, r_term, 
                            overnight_domestic_rate, overnight_foreign_rate, spot_price, long_thresh, short_thresh, signal_strength, length_option=1/12,
                            initial_capital=1_000_000 , signal_multiplier=1, signal_lb = 0.01, signal_ub = 0.99, plot=False, model_name='BMSM Backtest', 
                            base_notional=1_000_000, max_directional_risk=500, transaction_costs_indicator=True, 
                            transaction_costs_spot=0.0002 / 2, transaction_costs_option=0.0005 / 2, verbose = True, convert_USD = True):

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
        overnight_foreign_rate=overnight_foreign_rate,
        signal_strength=signal_strength,
        signal_multiplier=signal_multiplier,
        signal_strength_lb=signal_lb,
        signal_strength_up=signal_ub,
        base_notional_option=base_notional, 
        directional_risk_max=max_directional_risk, 
        transaction_cost_bool=transaction_costs_indicator, 
        transaction_costs_spot=transaction_costs_spot, 
        transaction_costs_option=transaction_costs_option,
        verbose=verbose,
        convert_USD=convert_USD
    )

    cash, trading_portfolio_value, total_portfolio_value, trading_summary, option_delta_positions, delta_hedge, delta_hedge_cashflows, hedge_carry, daily_option_pnl_real, daily_option_pnl_mtm, daily_hedge_pnl_mtm = strategy.run_strategy()
    cash_no_trades = strategy.compute_portfolio_value_with_no_trades()
    sharpe, max_dd, calmar, cagr, precision_all, precision_active, win_loss_ratio_all, win_loss_ratio_active, avg_win_all, avg_win_active, avg_loss_all, avg_loss_active, non_active_share, total_trades = strategy.performance_summary()

    if plot:

        plt.style.use('seaborn-v0_8-dark')
        plt.figure(figsize=(14, 6))
        plt.plot(cash, label='Cash with Trading', color='blue')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Cash Value", fontsize=16)
        plt.title(f"Total Cash over Trading Period (USD) - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.plot(daily_option_pnl_mtm.cumsum(), label='Option PnL', color='blue')
        plt.plot(daily_hedge_pnl_mtm.cumsum(), label='Hedge PnL (from hedge position)', color='orange')
        plt.plot(hedge_carry.cumsum(), label='Hedge PnL (from carry)', color='green')
        plt.plot((daily_option_pnl_mtm + daily_hedge_pnl_mtm + hedge_carry).cumsum(), label='Total PnL', color='red')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Trading PnL", fontsize=16)
        plt.title(f"Cumulative Mark to Market Trading PnL (USD) - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.plot(daily_option_pnl_mtm.cumsum(), label='Option PnL MtM', color='orange')
        plt.plot(daily_option_pnl_real.cumsum(), label='Option PnL Realised', color='blue')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Trading PnL", fontsize=16)
        plt.title(f"Cumulative Option Trading PnL (USD) - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.plot(total_portfolio_value, label='Total Portfolio Value', color='green')
        plt.xlabel("Day", fontsize=16)
        plt.ylabel("Total Portfolio Value", fontsize=16)
        plt.title(f"Total Mark to Market Portfolio Value Over Time (USD) - {model_name}", fontsize=22)
        plt.legend(fontsize = 'xx-large')
        plt.grid()
        plt.tight_layout()

    return cash, trading_portfolio_value, total_portfolio_value, trading_summary, option_delta_positions, delta_hedge, delta_hedge_cashflows, hedge_carry, daily_option_pnl_real, daily_option_pnl_mtm, daily_hedge_pnl_mtm, sharpe, max_dd, calmar, cagr, cash_no_trades, \
    precision_all, precision_active, win_loss_ratio_all, win_loss_ratio_active, avg_win_all, avg_win_active, avg_loss_all, avg_loss_active, non_active_share, total_trades


class Hedging_Trading_Strategy:

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
                overnight_foreign_rate: np.ndarray,
                signal_strength: np.ndarray, 
                signal_multiplier: float, 
                signal_strength_lb: float, 
                signal_strength_up: float, 
                base_notional_option: int,    # max amount to trade
                directional_risk_max: int, 
                transaction_costs_spot: float,
                transaction_costs_option: float, 
                transaction_cost_bool: bool, 
                verbose: bool, 
                convert_USD: bool
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
        self.overnight_foreign_rate = overnight_foreign_rate
        self.signal_strength = signal_strength
        self.signal_multiplier = signal_multiplier
        self.signal_strength_lb = signal_strength_lb
        self.signal_strength_up = signal_strength_up
        self.base_notional_option = base_notional_option
        self.test_vol_smile = test_vol_smile
        self.directional_risk_max = directional_risk_max
        self.transaction_cost_bool = transaction_cost_bool
        self.verbose = verbose
        self.convert_USD = convert_USD

        if self.transaction_cost_bool:
            self.transaction_costs_spot = transaction_costs_spot
            self.transaction_costs_option = transaction_costs_option
        else:
            self.transaction_costs_spot = 0.0
            self.transaction_costs_option = 0.0

        self.trading_summary = {
            'Time': [],
            'Notional Adjusted': [],
            'Enter Option Trade Size': [],
            'Strike Prices': [],
            'Option Positions': [], 
            'Total Number of Trades': 0}
        
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

        # scale base notional by signal in [0,1] times by a backtest multiplier
        # this is in domestic currency > straddle in domestic and so is this
        notional_adjusted = self.base_notional_option * signal_strength_bounded * self.signal_multiplier 

        enter_trade_size = notional_adjusted * unit_straddle_value  # total cash value of the trade in domestic

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

        # convert to USD
        if self.convert_USD:
            rates = np.insert(self.spot_price, 0, self.spot_price[0]) # align with portfolio arrays
            V_no_trades_USD = V_no_trades / rates
            return V_no_trades_USD
        else:
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
    
    def get_transaction_costs_option_leg(self, trade_size):
        '''
        To start with compute transaction costs for option leg in simplest way (fixed percent of complete trade size).
        '''

        # times by spot so this is in domestic currency too
        transaction_cost = self.transaction_costs_option * trade_size

        return transaction_cost

    def update_portfolio_open_positions(self, idx, trade_size, type):

        if type == 'long':
            sign = 1
        elif type =='short':
            sign = -1

        self.hedge_carry[idx] = 0.0  # no carry at start
        self.option_delta_position[idx] = 0.0 # delta neutral at start for ATM straddle
        self.total_hedge_position[idx] = 0.0  # no hedge position at start
        self.total_adjustments[idx] = 0.0 
        self.adjustment_cashflow[idx] = 0.0

        adjusted_trade_size = trade_size * sign

        self.MM_V[idx] -= adjusted_trade_size  # deduct cost of trade (already in domestic) from cash (if long then minus otherwise will add)
        self.trading_V[idx] = adjusted_trade_size

        self.daily_hedge_pnl_mtm[idx] = 0.0
        self.option_mtm_value[idx] = adjusted_trade_size
        self.daily_option_pnl_mtm[idx] = 0.0
        self.daily_option_pnl_real[idx] = 0.0 

        trans_cost = self.get_transaction_costs_option_leg(trade_size) # always use pos value
        self.MM_V[idx] -= trans_cost

        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

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

        self.update_portfolio_open_positions(
            idx=idx,
            trade_size=total_trade_size,
            type='long')

        if self.verbose:
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

        self.update_portfolio_open_positions(
            idx=idx,
            trade_size=total_trade_size,
            type='short')
        
        if self.verbose:
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

            vol = max(1e-8, vol_smile_t.iloc[i])

            if np.isnan(delta):
                inv_term = np.clip(np.exp(r_b_t * tau) * -0.5 + 1, 1e-8, 1-1e-8)
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strike_put = spot_t / np.exp(exp_term)

                inv_term = np.exp(r_b_t * tau) * 0.5
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strike_call = spot_t / np.exp(exp_term)

                strikes[i] = (strike_put + strike_call) / 2  


            elif delta < 0:
                inv_term = np.clip(np.exp(r_b_t * tau) * delta + 1, 1e-8, 1-1e-8)
                term1 = norm.ppf(inv_term)
                exp_term = (term1 * vol * np.sqrt(tau)) - ((r_t_t - r_b_t + 0.5 * vol**2) * tau)
                strikes[i] = spot_t / np.exp(exp_term)
            
            elif delta > 0:
                inv_term = np.clip(np.exp(r_b_t * tau) * delta, 1e-8, 1-1e-8)
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

    def get_transaction_costs_spot(self, spot, hedge_adjustments):
        '''
        Computes cost as cost * S_t * abs(hedge adjustments).
        '''
        
        transaction_cost = self.transaction_costs_spot * spot * abs(hedge_adjustments)

        return transaction_cost

    def update_portfolio_close_positions(self, idx, enter_trade_size, exit_trade_size, spot_t, type):

        if type =='long':
            sign =1
        elif type == 'short':
            sign = -1

        adjusted_exit_trade_size = exit_trade_size * sign
        adjusted_enter_trade_size = enter_trade_size * sign

        # exit the hedge position as well
        self.option_delta_position[idx] = 0.0

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)
        self.total_hedge_position[idx] = - self.option_delta_position[idx]
        self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx - 1]) 
        self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]

        self.trading_V[idx] = 0.0
        self.MM_V[idx] += adjusted_exit_trade_size + self.adjustment_cashflow[idx]  

        option_pnl = adjusted_exit_trade_size - adjusted_enter_trade_size
        self.daily_option_pnl_real[idx] = option_pnl
        self.option_mtm_value[idx] = adjusted_exit_trade_size
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 

        trans_cost_options = self.get_transaction_costs_option_leg(exit_trade_size)
        trans_cost_spot = self.get_transaction_costs_spot(spot_t, self.total_adjustments[idx])
        total_transaction_cost = trans_cost_options + trans_cost_spot
        self.MM_V[idx] -= total_transaction_cost

        self.V[idx] = self.MM_V[idx] + self.trading_V[idx]

        return option_pnl
        
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
        
        option_pnl = self.update_portfolio_close_positions(
            idx=idx,
            enter_trade_size=last_enter_trade_size,
            exit_trade_size=exit_trade_size,
            spot_t=spot_t,
            type='long')

        if self.verbose:
            print(f"Time {idx}: Long straddle exited with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")
            print(f"Option PnL for this trade: {option_pnl:.2f}")

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
        
        option_pnl = self.update_portfolio_close_positions(
            idx=idx,
            enter_trade_size=last_enter_trade_size,
            exit_trade_size=exit_trade_size,
            spot_t=spot_t,
            type='short')

        if self.verbose:
            print(f"Time {idx}: Short straddle exited with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")
            print(f"Option PnL for this trade: {option_pnl:.2f}")
        
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

        option_pnl = self.update_portfolio_close_positions(
            idx=idx,
            enter_trade_size=last_enter_trade_size,
            exit_trade_size=exit_trade_size,
            spot_t=spot_t,
            type='long')

        if self.verbose:
            print(f"Time {idx}: Long straddle expired with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")
            print(f"Option PnL for this trade: {option_pnl:.2f}")
        
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

        option_pnl = self.update_portfolio_close_positions(
            idx=idx,
            enter_trade_size=last_enter_trade_size,
            exit_trade_size=exit_trade_size,
            spot_t=spot_t,
            type='short')

        if self.verbose:
            print(f"Time {idx}: Short straddle expired with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")
            print(f"Option PnL for this trade: {option_pnl:.2f}")
        
        self.trading_summary['Enter Option Trade Size'].append(0.0)  # no trade size for exit
        self.trading_summary['Strike Prices'].append(np.nan)  # no strike price for exit
        self.trading_summary['Notional Adjusted'].append(0.0)  # no notional adjusted for exit

        return position
    
    def update_portfolio_mtm_positions(self, idx, delta_size, mtm_size, spot_t, type):

        if type == 'long':
            sign = 1
        elif type == 'short':
            sign = -1

        delta_adjusted = delta_size * sign
        mtm_size_adjusted = mtm_size * sign

        self.option_delta_position[idx] = delta_adjusted

        self.hedge_carry[idx] = self.total_hedge_position[idx - 1] * self.spot_price[idx-2] * self.r_base.iloc[idx-2] * (1/360)

        hedge_diff = (-self.option_delta_position[idx] - self.total_hedge_position[idx-1])

        if np.abs(hedge_diff) > self.directional_risk_max:

            self.total_hedge_position[idx] = - self.option_delta_position[idx] # hedge is negative of position delta
            self.total_adjustments[idx] = (self.total_hedge_position[idx] - self.total_hedge_position[idx-1])  
            self.adjustment_cashflow[idx] = -self.total_adjustments[idx] * spot_t + self.hedge_carry[idx]
        
        else:

            self.total_hedge_position[idx] = self.total_hedge_position[idx-1]  # no change in hedge position
            self.total_adjustments[idx] = 0
            self.adjustment_cashflow[idx] = self.hedge_carry[idx]

        self.trading_V[idx] = mtm_size_adjusted + (self.total_hedge_position[idx] * spot_t)
        self.MM_V[idx] += self.adjustment_cashflow[idx]   

        self.daily_option_pnl_real[idx] = 0.0

        self.option_mtm_value[idx] = mtm_size_adjusted
        self.daily_option_pnl_mtm[idx] = self.option_mtm_value[idx] - self.option_mtm_value[idx-1]

        self.daily_hedge_pnl_mtm[idx] = self.total_hedge_position[idx-1] * (spot_t - self.spot_price[idx-2]) 

        trans_cost_spot = self.get_transaction_costs_spot(spot_t, self.total_adjustments[idx])
        self.MM_V[idx] -= trans_cost_spot
        self.V[idx] = self.MM_V[idx] + self.trading_V[idx] 

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

        self.update_portfolio_mtm_positions(
            idx = idx, 
            delta_size=delta_size,
            mtm_size=mtm_trade_size,
            spot_t=spot_t,
            type = 'long')

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

        self.update_portfolio_mtm_positions(
            idx = idx, 
            delta_size=delta_size,
            mtm_size=mtm_trade_size,
            spot_t=spot_t,
            type = 'short')

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
        
        option_pnl = self.update_portfolio_close_positions(
            idx=idx,
            enter_trade_size=last_enter_trade_size,
            exit_trade_size=exit_trade_size,
            spot_t=spot_t,
            type = 'long')

        if self.verbose:
            print(f"Time {idx}: Long straddle exited (end of session) with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")
            print(f"Option PnL for this trade: {option_pnl:.2f}")
        
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
        
        option_pnl = self.update_portfolio_close_positions(
            idx=idx,
            enter_trade_size=last_enter_trade_size,
            exit_trade_size=exit_trade_size,
            spot_t=spot_t,
            type = 'short')

        if self.verbose:
            print(f"Time {idx}: Short straddle exited (end of session) with trade size {exit_trade_size:.2f}, notional adjusted {last_notional_adjusted:.2f}. Total cash available {self.MM_V[idx]:.2f}")
            print(f"Option PnL for this trade: {option_pnl:.2f}")
        
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
            signal_strength_t = self.signal_strength[t-1] 
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

        if self.verbose:
            print(f"Ensure no long trades active at the end of the trading period: {position == 0}")
            print(f"Final cash available after all trades: {self.MM_V[-1]:.6f}")

        if self.convert_USD:
            self.convert_to_USD()
            return self.MM_V_USD, self.trading_V_USD, self.V_USD, self.trading_summary, self.option_delta_position, self.total_hedge_position, self.adjustment_cashflow_USD, self.hedge_carry_USD, self.daily_option_pnl_real_USD, self.daily_option_pnl_mtm_USD, self.daily_hedge_pnl_mtm_USD
        else:
            return self.MM_V, self.trading_V, self.V, self.trading_summary, self.option_delta_position, self.total_hedge_position, self.adjustment_cashflow, self.hedge_carry, self.daily_option_pnl_real, self.daily_option_pnl_mtm, self.daily_hedge_pnl_mtm
    
    def convert_to_USD(self):

        rates = np.insert(self.spot_price, 0, self.spot_price[0]) # align with portfolio arrays
        self.MM_V_USD = self.MM_V / rates
        self.trading_V_USD = self.trading_V / rates
        self.V_USD = self.V /rates
        self.adjustment_cashflow_USD = self.adjustment_cashflow / rates
        self.hedge_carry_USD = self.hedge_carry / rates
        self.daily_option_pnl_real_USD = self.daily_option_pnl_real / rates
        self.daily_option_pnl_mtm_USD = self.daily_option_pnl_mtm / rates
        self.daily_hedge_pnl_mtm_USD = self.daily_hedge_pnl_mtm / rates

    def performance_summary(self):

        if self.convert_USD:
            V = self.V_USD # USD currency
            risk_free_rate = self.overnight_foreign_rate.values
        else:
            V = self.V # domestic currency (ie already USD)
            risk_free_rate = self.overnight_domestic_rate.values

        if self.verbose:
            print("\n")
            print("Performance Summary (in USD):") 

        # get portfolio returns
        returns_V = (V[1:] - V[:-1]) / V[:-1]
        excess_returns = returns_V - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        sharpe_annualised = sharpe * np.sqrt(252)  

        if self.verbose:
            print(f"Sharpe Ratio (Annualised): {sharpe_annualised:.6f}")

        rolling_max = np.maximum.accumulate(V)
        daily_drawdown = V / rolling_max - 1
        max_daily_drawdown = np.minimum.accumulate(daily_drawdown) # returns min value seen so far

        if self.verbose:
            print(f"Max Daily Drawdown: {max_daily_drawdown.min() * 100:.6f}%")

        calmar = np.mean(excess_returns) / max(-max_daily_drawdown.min(), 1e-10) * 252

        if self.verbose:
            print(f"Calmar Ratio (annualised): {calmar:.6f}")

        if self.verbose:
            print(f"Daily standard deviation of excess returns (annualised): {np.std(excess_returns) * np.sqrt(252):.6f}")

        # CAGR
        cagr = (V[-1] / V[0]) ** (1 / (len(V) / 252)) - 1

        if self.verbose:
            print(f"CAGR: {cagr*100:.6f}%")

        # precision - number of pos pnl moves above risk free rate
        all_positives = np.sum(excess_returns > 0)
        precision_all = all_positives / len(excess_returns) if len(excess_returns) > 0 else 0

        active_mask = (np.abs(self.trading_summary['Option Positions'][1:]) > 0)  
        active_excess = excess_returns[active_mask]

        active_positives = np.sum(active_excess > 0)
        precision_active = active_positives / len(active_excess) if len(active_excess) > 0 else 0

        non_active_share = 1.0 - np.mean(active_mask)

        # total number trades
        total_trades = self.trading_summary['Total Number of Trades']

        # win / loss rates all
        wins_all = excess_returns[excess_returns > 0]
        losses_all = excess_returns[excess_returns < 0]

        avg_win_all = np.mean(wins_all) if len(wins_all) > 0 else 0
        avg_loss_all = np.mean(losses_all) if len(losses_all) > 0 else 0

        win_loss_ratio_all = np.abs(avg_win_all / avg_loss_all) if avg_loss_all != 0 else np.inf

        # win / loss rates active
        wins_active = active_excess[active_excess > 0]
        losses_active = active_excess[active_excess < 0]

        avg_win_active = np.mean(wins_active) if len(wins_active) > 0 else 0
        avg_loss_active = np.mean(losses_active) if len(losses_active) > 0 else 0

        win_loss_ratio_active = np.abs(avg_win_active / avg_loss_active) if avg_loss_active != 0 else np.inf

        if self.verbose:
             print(f"Precision (All): {precision_all:.6f}, Precision (Active): {precision_active:.6f}")
             print(f"Win/Loss Ratio (All): {win_loss_ratio_all:.6f}, Win/Loss Ratio (Active): {win_loss_ratio_active:.6f}")
             print(f"Average Win (All): {avg_win_all*100:.6f}%, Average Win (Active): {avg_win_active*100:.6f}%")
             print(f"Average Loss (All): {avg_loss_all*100:.6f}%, Average Loss (Active): {avg_loss_active*100:.6f}%")
             print(f"Non-Active Share: {non_active_share:.6f}")
             print(f"Total Trades: {total_trades:.6f}")
             print("\n")

        return sharpe_annualised, max_daily_drawdown.min() * 100, calmar, cagr, precision_all, precision_active, win_loss_ratio_all, win_loss_ratio_active, avg_win_all, avg_win_active, avg_loss_all, avg_loss_active, non_active_share, total_trades

class Compare_Trading_Strategies:
        
        def __init__(self, 
                return_series: pd.DataFrame,  # as decimals
                realised_variance_series: pd.DataFrame, # as decimals
                atm_implied_vol_data: pd.DataFrame, # as percent
                vol_smile_data: pd.DataFrame,  # as percent
                train_size: float, 
                ticker: str, 
                initial_capital_domestic: float,
                notional_base: float, #ie in foreign currency
                maximum_delta_difference: float, 
                signal_lb: float,
                signal_ub: float, 
                transaction_cost_indicator: bool, 
                transaction_costs_spot: float,  # as decimal
                transaction_costs_option: float,    # as decimal
                option_maturity: float, # in years
                forecast_horizon: int, 
                spot_series: pd.DataFrame, 
                overnight_domestic_rate: pd.DataFrame,  # as percent - already test aligned same as below
                overnight_foreign_rate:pd.DataFrame,  # as percent - already test aligned same as below - needed for sharpes in USD
                domestic_rate: pd.DataFrame,   # as percent
                foreign_rate: pd.DataFrame,   # as percent
                plots: bool, 
                verbose: bool, 
                sort_hyperparams_by: str, 
                garch_1_2_indicator: bool,   # whether to use GARCH(1,2) model
                kbar: int, 
                b: float,
                gamma_kbar: float, 
                convert_USD: bool,
                M: int, # figarch lags
                OLS_INDICATOR: bool, 
                long_thresholds: np.ndarray,
                short_thresholds: np.ndarray,
                sig_multipliers: np.ndarray,
                alpha_implieds: np.ndarray
                ):

                self.return_series = return_series
                self.realised_variance_series = realised_variance_series
                self.atm_implied_vol_data = atm_implied_vol_data
                self.train_size = train_size
                self.test_size = 1 - train_size
                self.ticker = ticker
                self.initial_capital_domestic = initial_capital_domestic
                self.notional_base = notional_base
                self.maximum_delta_difference = maximum_delta_difference
                self.signal_lb = signal_lb
                self.signal_ub = signal_ub
                self.transaction_cost_indicator = transaction_cost_indicator
                self.transaction_costs_spot = transaction_costs_spot
                self.transaction_costs_option = transaction_costs_option
                self.option_maturity = option_maturity
                self.H = forecast_horizon
                self.spot_series = spot_series
                self.vol_smile_data = vol_smile_data
                self.spot_series = spot_series
                self.overnight_domestic_rate = overnight_domestic_rate
                self.overnight_foreign_rate = overnight_foreign_rate
                self.r_t = domestic_rate
                self.r_b = foreign_rate
                self.plots = plots
                self.verbose = verbose
                self.sort_hyperparams_by = sort_hyperparams_by
                self.garch_1_2_indicator = garch_1_2_indicator
                self.kbar = kbar
                self.b = b
                self.gamma_kbar = gamma_kbar
                self.convert_USD = convert_USD
                self.M = M
                self.OLS_INDICATOR = OLS_INDICATOR
                self.long_thresh_cv = long_thresholds
                self.short_thresh_cv = short_thresholds
                self.sig_multipliers_cv = sig_multipliers
                self.alpha_implied_cv = alpha_implieds

                self.N = self.return_series.shape[0]
                self.N_test = self.N // 2 - self.H

        def prepare_universal_series(self):

                self.train_realised_variance_backtest = self.realised_variance_series.values[1: self.N//2] # ignore first as pre-whitening shifts data
                self.train_realised_variance_oos = self.realised_variance_series.values[self.N//2 + int(self.N_test * self.train_size):]
                self.realised_variance_test_backtest = self.realised_variance_series.values[self.N//2:]
                self.realised_variance_test_oos = self.realised_variance_series.values[self.N//2 + int(self.N_test * self.train_size):]

                self.atm_mid_implied_vol_test = self.atm_implied_vol_data['Mid'].values[self.N//2:-self.H] / 100
                self.mid_vol_smile_test = self.vol_smile_data[self.N//2:-self.H] / 100
                self.S = self.spot_series['Mid'].values

                self.atm_mid_implied_vol_backtest = self.atm_mid_implied_vol_test[:int(self.N_test * self.train_size)]
                self.r_b_backtest = self.r_b[:int(self.N_test * self.train_size)]
                self.r_t_backtest = self.r_t[:int(self.N_test * self.train_size)]
                self.overnight_dom_r_backtest = self.overnight_domestic_rate[:int(self.N_test * self.train_size)]
                self.overnight_for_r_backtest = self.overnight_foreign_rate[:int(self.N_test * self.train_size)]
                self.spot_price_backtest = self.S[:int(self.N_test * self.train_size)]
                self.vol_smile_backtest = self.mid_vol_smile_test[:int(self.N_test * self.train_size)]

                self.atm_mid_implied_vol_oos = self.atm_mid_implied_vol_test[int(self.N_test * self.train_size):]
                self.r_b_oos = self.r_b[int(self.N_test * self.train_size):]
                self.r_t_oos = self.r_t[int(self.N_test * self.train_size):]
                self.overnight_dom_r_oos = self.overnight_domestic_rate[int(self.N_test * self.train_size):]
                self.overnight_for_r_oos = self.overnight_foreign_rate[int(self.N_test * self.train_size):]
                self.spot_price_oos = self.S[int(self.N_test * self.train_size):]
                self.vol_smile_oos = self.mid_vol_smile_test[int(self.N_test * self.train_size):]

                self.e_data, _, _ = pre_whitening(self.return_series.values.flatten())
                self.e_train_backtest = self.e_data[:int(self.N * 0.5) - 1]  # -1 to match the length of test_realised_variances
                self.e_test_backtest = self.e_data[int(self.N * 0.5) - 1:]

                self.e_train_oos = self.e_data[:int(self.N * 0.5) - 1 + int(self.N_test * self.train_size)]
                self.e_test_oos = self.e_data[int(self.N * 0.5) - 1 + int(self.N_test * self.train_size):]

        def get_BMSM_data(self):

                initial_params_backtest = np.array([1.3, np.std(self.e_train_backtest)*100])

                bmsm_forecasts_backtest = BMSM_Forecaster_MLE_delta_stds(
                        initial_params=initial_params_backtest,
                        train_data=self.e_train_backtest,
                        test_data=self.e_test_backtest,
                        train_realised_variance=self.train_realised_variance_backtest.flatten(),
                        test_realised_variance=self.realised_variance_test_backtest.flatten(), 
                        kbar=self.kbar,
                        H=self.H, 
                        scale=100, 
                        b = self.b, 
                        gamma_kbar=self.gamma_kbar,
                        percent_space=True)

                norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_vols, realised_vol_H_forecasts_train, realised_vol_H_true_train = bmsm_forecasts_backtest.compute_forecasts()

                if self.verbose:
                        print(f"\nBMSM backtest estimation results:")
                        print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")

                # get decimal forecasts that line up with implied vols (1: so signal is based on our forecast for today vs implied vol yday)
                realised_vol_forecasts = realised_vol_H_forecasts[1:] / 100
                self.forecasts_bmsm_backtest = realised_vol_forecasts[:int(self.N_test * self.train_size)]

                stds_vol = stds_of_vols[1:]
                stds_vol_backtest = stds_vol[:int(self.N_test * self.train_size)]
                signal_backtest = 1 / stds_vol_backtest  
                self.signal_bmsm_backtest = (signal_backtest - signal_backtest.min()) / (signal_backtest.max() - signal_backtest.min())

                self.true_vol_series = realised_vol_H_true[1:] / 100

                self.forecasts_train_ols_bmsm_backtest = realised_vol_H_forecasts_train / 100
                self.true_vol_series_train_ols_bmsm_backtest = realised_vol_H_true_train / 100 

                # re fit data completely for OOS testing
                initial_params_oos = np.array([1.3, np.std(self.e_train_oos)*100])

                bmsm_forecasts_oos = BMSM_Forecaster_MLE_delta_stds(
                        initial_params=initial_params_oos,
                        train_data=self.e_train_oos,
                        test_data=self.e_test_oos,
                        train_realised_variance=self.train_realised_variance_oos.flatten(),
                        test_realised_variance=self.realised_variance_test_oos.flatten(), 
                        kbar=self.kbar,
                        H=self.H, 
                        scale=100, 
                        b=self.b,
                        gamma_kbar=self.gamma_kbar,
                        percent_space=True)
                
                norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_vols, realised_vol_H_forecasts_train, realised_vol_H_true_train = bmsm_forecasts_oos.compute_forecasts()
                
                if self.verbose:
                        print(f"\nBMSM out of sample estimation results:")
                        print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")

                # get decimal forecasts that line up with implied vols - again one forward so no forward look when comparing to implied vols
                self.forecasts_bmsm_oos = realised_vol_H_forecasts[1:] / 100
                signal_oos = 1 / stds_of_vols[1:]  

                # we want to normalise signals without adding forward look bias
                bt_min = signal_backtest.min()
                bt_max = signal_backtest.max()

                norm_signal_oos = []
                cur_min, cur_max = bt_min, bt_max

                for val in signal_oos:

                    norm_signal_oos.append((val - cur_min) / (cur_max - cur_min))

                    cur_min = min(cur_min, val)
                    cur_max = max(cur_max, val)

                norm_signal_oos = np.array(norm_signal_oos)
                self.signal_bmsm_oos = norm_signal_oos       

                self.forecasts_train_ols_bmsm_oos = realised_vol_H_forecasts_train / 100

        def get_hyperparameters(self, forecasts_backtest, signal_backtest, forecasts_train_ols, true_train_ols):

                backtest_data = [forecasts_backtest, 
                 self.atm_mid_implied_vol_backtest, 
                 self.r_b_backtest,
                 self.r_t_backtest,
                 self.overnight_dom_r_backtest, 
                 self.spot_price_backtest, 
                 self.vol_smile_backtest, 
                 signal_backtest, 
                 self.overnight_for_r_backtest, 
                 forecasts_train_ols, 
                 self.atm_implied_vol_data['Mid'].values[:len(forecasts_train_ols)], 
                 true_train_ols]

                param_grid = {
                'long_threshold': self.long_thresh_cv,
                'short_threshold': self.short_thresh_cv,
                'signal_multiplier': self.sig_multipliers_cv, 
                'alpha_implied': self.alpha_implied_cv}

                cv_results = grid_search_params(backtest_data, param_grid, self.initial_capital_domestic, self.option_maturity, self.signal_lb, self.signal_ub, 
                                                self.notional_base, self.maximum_delta_difference, self.transaction_cost_indicator, self.transaction_costs_spot, self.transaction_costs_option, verbose = False, convert_USD=self.convert_USD, OLS = self.OLS_INDICATOR)
                
                best_results = cv_results.sort_values(by=self.sort_hyperparams_by, ascending=False)

                long_optimal = best_results['long_threshold'].iloc[0]
                short_optimal = best_results['short_threshold'].iloc[0]
                sig_mult = best_results['signal_multiplier'].iloc[0]
                alpha_implied = best_results['alpha_implied'].iloc[0]

                return long_optimal, short_optimal, sig_mult, alpha_implied

        def run_trading_strategy(self, forecasts, signal, long_optimal, short_optimal, sig_mult, type, model, plots, verbose):
                
                if type == 'Backtest':
                        cash_bt, trading_portfolio_value_bt, total_portfolio_value_bt, \
                        trading_summary_bt, option_delta_positions_bt, delta_hedge_bt, \
                        delta_hedge_cashflows_bt, hedge_carry_bt, daily_option_pnl_real_bt, \
                        daily_option_pnl_mtm_bt,daily_hedge_pnl_mtm_bt,sharpe_bt, max_dd_bt, \
                        calmar_bt, cagr_bt, cash_no_trades, precision_all_bt, precision_active_bt, \
                        win_loss_ratio_all_bt, win_loss_ratio_active_bt, avg_win_all_bt, avg_win_active_bt, \
                        avg_loss_all_bt, avg_loss_active_bt, non_active_share_bt, total_trades_bt = run_strategy_with_params(
                                forecasts=forecasts,
                                test_implied_vol=self.atm_mid_implied_vol_backtest,
                                test_vol_smile=self.vol_smile_backtest,
                                r_base=self.r_b_backtest,
                                r_term=self.r_t_backtest,
                                overnight_domestic_rate=self.overnight_dom_r_backtest,
                                overnight_foreign_rate=self.overnight_for_r_backtest,
                                spot_price=self.spot_price_backtest,
                                long_thresh=long_optimal,
                                short_thresh=short_optimal,
                                signal_strength=signal,
                                length_option=self.option_maturity,
                                initial_capital=self.initial_capital_domestic,
                                signal_multiplier=sig_mult,
                                signal_lb=self.signal_lb,
                                signal_ub=self.signal_ub,
                                plot=plots,
                                model_name=model + " " + type, 
                                base_notional=self.notional_base, 
                                max_directional_risk=self.maximum_delta_difference, 
                                transaction_costs_indicator=self.transaction_cost_indicator,
                                transaction_costs_spot=self.transaction_costs_spot,
                                transaction_costs_option=self.transaction_costs_option, 
                                verbose = verbose, 
                                convert_USD=self.convert_USD)

                        return cash_bt, trading_portfolio_value_bt, total_portfolio_value_bt, \
                        trading_summary_bt, option_delta_positions_bt, delta_hedge_bt, \
                        delta_hedge_cashflows_bt, hedge_carry_bt, daily_option_pnl_real_bt, \
                        daily_option_pnl_mtm_bt,daily_hedge_pnl_mtm_bt,sharpe_bt, max_dd_bt, \
                        calmar_bt, cagr_bt, cash_no_trades, precision_all_bt, precision_active_bt, \
                        win_loss_ratio_all_bt, win_loss_ratio_active_bt, avg_win_all_bt, avg_win_active_bt, \
                        avg_loss_all_bt, avg_loss_active_bt, non_active_share_bt, total_trades_bt
                
                elif type == 'Out of Sample':
                        cash_oos, trading_portfolio_value_oos, total_portfolio_value_oos, \
                        trading_summary_oos, option_delta_positions_oos, delta_hedge_oos, \
                        delta_hedge_cashflows_oos, hedge_carry_oos, daily_option_pnl_real_oos, \
                        daily_option_pnl_mtm_oos,daily_hedge_pnl_mtm_oos,sharpe_oos, max_dd_oos, \
                        calmar_oos, cagr_oos, cash_no_trades, precision_all_oos, precision_active_oos, \
                        win_loss_ratio_all_oos, win_loss_ratio_active_oos, avg_win_all_oos, avg_win_active_oos, \
                        avg_loss_all_oos, avg_loss_active_oos, non_active_share_oos, total_trades_oos = run_strategy_with_params(
                                forecasts=forecasts,
                                test_implied_vol=self.atm_mid_implied_vol_oos,
                                test_vol_smile=self.vol_smile_oos,
                                r_base=self.r_b_oos,
                                r_term=self.r_t_oos,
                                overnight_domestic_rate=self.overnight_dom_r_oos,
                                overnight_foreign_rate=self.overnight_for_r_oos,
                                spot_price=self.spot_price_oos,
                                long_thresh=long_optimal,
                                short_thresh=short_optimal,
                                signal_strength=signal,
                                length_option=self.option_maturity,
                                initial_capital=self.initial_capital_domestic,
                                signal_multiplier=sig_mult,
                                signal_lb=self.signal_lb,
                                signal_ub=self.signal_ub,
                                plot=plots,
                                model_name=model + " " + type, 
                                base_notional=self.notional_base, 
                                max_directional_risk=self.maximum_delta_difference, 
                                transaction_costs_indicator=self.transaction_cost_indicator,
                                transaction_costs_spot=self.transaction_costs_spot,
                                transaction_costs_option=self.transaction_costs_option, 
                                verbose = verbose, 
                                convert_USD=self.convert_USD)

                        return cash_oos, trading_portfolio_value_oos, total_portfolio_value_oos, \
                        trading_summary_oos, option_delta_positions_oos, delta_hedge_oos, \
                        delta_hedge_cashflows_oos, hedge_carry_oos, daily_option_pnl_real_oos, \
                        daily_option_pnl_mtm_oos,daily_hedge_pnl_mtm_oos,sharpe_oos, max_dd_oos, \
                        calmar_oos, cagr_oos, cash_no_trades, precision_all_oos, precision_active_oos, \
                        win_loss_ratio_all_oos, win_loss_ratio_active_oos, avg_win_all_oos, avg_win_active_oos, \
                        avg_loss_all_oos, avg_loss_active_oos, non_active_share_oos, total_trades_oos 

                else:
                   print("Invalid type specified. Please use 'Backtest' or 'Out of Sample'.")
                   return None

        def get_BMSM_hyperparams_and_run_strategy(self, verbose):

                self.get_BMSM_data()

                self.long_optimal_bmsm, self.short_optimal_bmsm, self.sig_mult_bmsm, self.alpha_implied_bmsm = \
                        self.get_hyperparameters(self.forecasts_bmsm_backtest, self.signal_bmsm_backtest, self.forecasts_train_ols_bmsm_backtest, \
                                                 self.true_vol_series_train_ols_bmsm_backtest)
                
                self.name_bmsm = 'BMSM'

                if self.OLS_INDICATOR:
                    self.forecasts_bmsm_backtest, self.smear_bmsm, self.beta_bmsm, self.scalar_bmsm = regression_forecasts_ridge_train(self.forecasts_train_ols_bmsm_backtest, \
                                                                 self.atm_implied_vol_data['Mid'].values[:len(self.forecasts_train_ols_bmsm_backtest)], self.alpha_implied_bmsm, \
                                                                 self.true_vol_series_train_ols_bmsm_backtest, self.forecasts_bmsm_backtest, \
                                                                 self.atm_mid_implied_vol_backtest)
                    self.name_bmsm = 'BMSM with Implied'
                
                self.cash_bmsm_bt, self.trading_portfolio_value_bmsm_bt, self.total_portfolio_value_bmsm_bt, \
                        self.trading_summary_bmsm_bt, self.option_delta_positions_bmsm_bt, self.delta_hedge_bmsm_bt, \
                        self.delta_hedge_cashflows_bmsm_bt, self.hedge_carry_bmsm_bt, self.daily_option_pnl_real_bmsm_bt, \
                        self.daily_option_pnl_mtm_bmsm_bt,self.daily_hedge_pnl_mtm_bmsm_bt,self.sharpe_bmsm_bt, self.max_dd_bmsm_bt, \
                        self.calmar_bmsm_bt, self.cagr_bmsm_bt, _, self.precision_all_bmsm_bt, self.precision_active_bmsm_bt, \
                        self.win_loss_ratio_all_bmsm_bt, self.win_loss_ratio_active_bmsm_bt, self.avg_win_all_bmsm_bt, self.avg_win_active_bmsm_bt, \
                        self.avg_loss_all_bmsm_bt, self.avg_loss_active_bmsm_bt, self.non_active_share_bmsm_bt, self.total_trades_bmsm_bt = \
                                self.run_trading_strategy(self.forecasts_bmsm_backtest, self.signal_bmsm_backtest, \
                                                        self.long_optimal_bmsm, self.short_optimal_bmsm, self.sig_mult_bmsm, \
                                                        "Backtest", self.name_bmsm, self.plots, verbose)

                if self.OLS_INDICATOR:
                     self.forecasts_bmsm_oos = regression_forecasts_ridge_test(self.forecasts_bmsm_oos, \
                                                                 self.atm_mid_implied_vol_oos, self.smear_bmsm, \
                                                                 self.beta_bmsm, self.scalar_bmsm)

                self.cash_bmsm_oos, self.trading_portfolio_value_bmsm_oos, self.total_portfolio_value_bmsm_oos, \
                        self.trading_summary_bmsm_oos, self.option_delta_positions_bmsm_oos, self.delta_hedge_bmsm_oos, \
                        self.delta_hedge_cashflows_bmsm_oos, self.hedge_carry_bmsm_oos, self.daily_option_pnl_real_bmsm_oos, \
                        self.daily_option_pnl_mtm_bmsm_oos,self.daily_hedge_pnl_mtm_bmsm_oos,self.sharpe_bmsm_oos, self.max_dd_bmsm_oos, \
                        self.calmar_bmsm_oos, self.cagr_bmsm_oos, _, self.precision_all_bmsm_oos, self.precision_active_bmsm_oos, \
                        self.win_loss_ratio_all_bmsm_oos, self.win_loss_ratio_active_bmsm_oos, self.avg_win_all_bmsm_oos, self.avg_win_active_bmsm_oos, \
                        self.avg_loss_all_bmsm_oos, self.avg_loss_active_bmsm_oos, self.non_active_share_bmsm_oos, self.total_trades_bmsm_oos = \
                                self.run_trading_strategy(self.forecasts_bmsm_oos, self.signal_bmsm_oos, \
                                                        self.long_optimal_bmsm, self.short_optimal_bmsm, self.sig_mult_bmsm, \
                                                        "Out of Sample", self.name_bmsm, self.plots, verbose)

        def get_GARCH_data(self):

                if self.garch_1_2_indicator:

                        garch_forecaster = GARCH_Forecaster_MLE_p_1_q_2_delta_stds(
                                train_data=self.e_train_backtest,
                                test_data=self.e_test_backtest,
                                train_realised_variance=self.train_realised_variance_backtest.flatten(),
                                test_realised_variance=self.realised_variance_test_backtest.flatten(),
                                H=self.H, 
                                scale=100, 
                                percent_space=True)

                        res = garch_forecaster.fit_garch_model_library()
                        norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_vols, forecasts_train, true_vols_train = garch_forecaster.forecast_garch()

                        if self.verbose:
                                print(f"\n GARCH(1, 2) backtest estimation results:")
                                print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")


                        # get decimal forecasts that line up with implied vols (1: so signal is based on our forecast for today vs implied vol yday)
                        realised_vol_forecasts = realised_vol_H_forecasts[1:] / 100
                        self.forecasts_garch_backtest = realised_vol_forecasts[:int(self.N_test * self.train_size)]

                        stds_vol = stds_of_vols[1:]
                        stds_vol_backtest = stds_vol[:int(self.N_test * self.train_size)]
                        signal_backtest = 1 / stds_vol_backtest  
                        self.signal_garch_backtest = (signal_backtest - signal_backtest.min()) / (signal_backtest.max() - signal_backtest.min())

                        self.forecasts_train_ols_garch_backtest = forecasts_train / 100
                        self.true_vol_series_train_ols_garch_backtest = true_vols_train / 100

                        garch_forecaster = GARCH_Forecaster_MLE_p_1_q_2_delta_stds(
                                train_data=self.e_train_oos,
                                test_data=self.e_test_oos,
                                train_realised_variance=self.train_realised_variance_oos.flatten(),
                                test_realised_variance=self.realised_variance_test_oos.flatten(),
                                H=self.H, 
                                scale=100, 
                                percent_space=True)

                        res = garch_forecaster.fit_garch_model_library()
                        norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_vols, forecasts_train, _ = garch_forecaster.forecast_garch()

                        if self.verbose:
                                print(f"\n GARCH(1, 2) OOS estimation results:")
                                print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")


                        # get decimal forecasts that line up with implied vols (1: so signal is based on our forecast for today vs implied vol yday)
                        realised_vol_forecasts = realised_vol_H_forecasts[1:] / 100
                        self.forecasts_garch_oos = realised_vol_forecasts

                        stds_vol = stds_of_vols[1:]
                        stds_vol_oos = stds_vol
                        signal_oos = 1 / stds_vol_oos  

                        # we want to normalise signals without adding forward look bias
                        bt_min = signal_backtest.min()
                        bt_max = signal_backtest.max()

                        norm_signal_oos = []
                        cur_min, cur_max = bt_min, bt_max

                        for val in signal_oos:

                            norm_signal_oos.append((val - cur_min) / (cur_max - cur_min))

                            cur_min = min(cur_min, val)
                            cur_max = max(cur_max, val)

                        norm_signal_oos = np.array(norm_signal_oos)
                        self.signal_garch_oos = norm_signal_oos 

                        self.forecasts_train_ols_garch_oos = forecasts_train / 100

                else:

                        initial_params = np.array([1e-6, 0.15, 0.82]) # omega, alpha, beta

                        garch_forecaster = GARCH_Forecaster_MLE_1_1_delta_stds(
                                train_data=self.e_train_backtest,
                                test_data=self.e_test_backtest,
                                train_realised_variance=self.train_realised_variance_backtest.flatten(),
                                test_realised_variance=self.realised_variance_test_backtest.flatten(),
                                initial_params=initial_params,
                                H=self.H, 
                                scale=100, 
                                percent_space=True)

                        res = garch_forecaster.estimate_garch()
                        norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_vols, forecasts_train, true_train = garch_forecaster.forecast_garch()

                        if self.verbose:
                                print(f"\n GARCH(1, 1) backtest estimation results:")
                                print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")


                        # get decimal forecasts that line up with implied vols (1: so signal is based on our forecast for today vs implied vol yday)
                        realised_vol_forecasts = realised_vol_H_forecasts[1:] / 100
                        self.forecasts_garch_backtest = realised_vol_forecasts[:int(self.N_test * self.train_size)]

                        stds_vol = stds_of_vols[1:]
                        stds_vol_backtest = stds_vol[:int(self.N_test * self.train_size)]
                        signal_backtest = 1 / stds_vol_backtest  
                        self.signal_garch_backtest = (signal_backtest - signal_backtest.min()) / (signal_backtest.max() - signal_backtest.min())

                        self.forecasts_train_ols_garch_backtest = forecasts_train / 100
                        self.true_vol_series_train_ols_garch_backtest = true_train / 100

                        garch_forecaster = GARCH_Forecaster_MLE_1_1_delta_stds(
                                train_data=self.e_train_oos,
                                test_data=self.e_test_oos,
                                train_realised_variance=self.train_realised_variance_oos.flatten(),
                                test_realised_variance=self.realised_variance_test_oos.flatten(),
                                initial_params=initial_params,
                                H=self.H, 
                                scale=100, 
                                percent_space=True)

                        res = garch_forecaster.estimate_garch()
                        norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_vols, forecasts_train, _ = garch_forecaster.forecast_garch()

                        if self.verbose:
                                print(f"\n GARCH(1, 1) OOS estimation results:")
                                print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")


                        # get decimal forecasts that line up with implied vols (1: so signal is based on our forecast for today vs implied vol yday)
                        realised_vol_forecasts = realised_vol_H_forecasts[1:] / 100
                        self.forecasts_garch_oos = realised_vol_forecasts

                        stds_vol = stds_of_vols[1:]
                        stds_vol_oos = stds_vol
                        signal_oos = 1 / stds_vol_oos  

                        # we want to normalise signals without adding forward look bias
                        bt_min = signal_backtest.min()
                        bt_max = signal_backtest.max()

                        norm_signal_oos = []
                        cur_min, cur_max = bt_min, bt_max

                        for val in signal_oos:

                            norm_signal_oos.append((val - cur_min) / (cur_max - cur_min))

                            cur_min = min(cur_min, val)
                            cur_max = max(cur_max, val)

                        norm_signal_oos = np.array(norm_signal_oos)
                        self.signal_garch_oos = norm_signal_oos 

                        self.forecasts_train_ols_garch_oos = forecasts_train / 100

        def get_GARCH_hyperparams_and_run_strategy(self, verbose):

                self.get_GARCH_data()

                self.long_optimal_garch, self.short_optimal_garch, self.sig_mult_garch, self.alpha_implied_garch = \
                        self.get_hyperparameters(self.forecasts_garch_backtest, self.signal_garch_backtest, self.forecasts_train_ols_garch_backtest, \
                                                 self.true_vol_series_train_ols_garch_backtest)

                self.name_garch = 'GARCH'

                if self.OLS_INDICATOR:
                    self.forecasts_garch_backtest, self.smear_garch, self.beta_garch, self.scalar_garch = regression_forecasts_ridge_train(self.forecasts_train_ols_garch_backtest, \
                                                                 self.atm_implied_vol_data['Mid'].values[:len(self.forecasts_train_ols_garch_backtest)], self.alpha_implied_garch, \
                                                                 self.true_vol_series_train_ols_garch_backtest, self.forecasts_garch_backtest, \
                                                                 self.atm_mid_implied_vol_backtest)
                    self.name_garch = 'GARCH with Implied'

                self.cash_garch_bt, self.trading_portfolio_value_garch_bt, self.total_portfolio_value_garch_bt, \
                        self.trading_summary_garch_bt, self.option_delta_positions_garch_bt, self.delta_hedge_garch_bt, \
                        self.delta_hedge_cashflows_garch_bt, self.hedge_carry_garch_bt, self.daily_option_pnl_real_garch_bt, \
                        self.daily_option_pnl_mtm_garch_bt,self.daily_hedge_pnl_mtm_garch_bt,self.sharpe_garch_bt, self.max_dd_garch_bt, \
                        self.calmar_garch_bt, self.cagr_garch_bt, _, self.precision_all_garch_bt, self.precision_active_garch_bt, \
                        self.win_loss_ratio_all_garch_bt, self.win_loss_ratio_active_garch_bt, self.avg_win_all_garch_bt, self.avg_win_active_garch_bt, \
                        self.avg_loss_all_garch_bt, self.avg_loss_active_garch_bt, self.non_active_share_garch_bt, self.total_trades_garch_bt = \
                                self.run_trading_strategy(self.forecasts_garch_backtest, self.signal_garch_backtest, \
                                                        self.long_optimal_garch, self.short_optimal_garch, self.sig_mult_garch, \
                                                        "Backtest", self.name_garch, self.plots, verbose)
                
                if self.OLS_INDICATOR:
                    self.forecasts_garch_oos = regression_forecasts_ridge_test(self.forecasts_garch_oos, \
                                                                                self.atm_mid_implied_vol_oos, self.smear_garch, \
                                                                                self.beta_garch, self.scalar_garch)

                self.cash_garch_oos, self.trading_portfolio_value_garch_oos, self.total_portfolio_value_garch_oos, \
                        self.trading_summary_garch_oos, self.option_delta_positions_garch_oos, self.delta_hedge_garch_oos, \
                        self.delta_hedge_cashflows_garch_oos, self.hedge_carry_garch_oos, self.daily_option_pnl_real_garch_oos, \
                        self.daily_option_pnl_mtm_garch_oos,self.daily_hedge_pnl_mtm_garch_oos,self.sharpe_garch_oos, self.max_dd_garch_oos, \
                        self.calmar_garch_oos, self.cagr_garch_oos, _, self.precision_all_garch_oos, self.precision_active_garch_oos, \
                        self.win_loss_ratio_all_garch_oos, self.win_loss_ratio_active_garch_oos, self.avg_win_all_garch_oos, self.avg_win_active_garch_oos, \
                        self.avg_loss_all_garch_oos, self.avg_loss_active_garch_oos, self.non_active_share_garch_oos, self.total_trades_garch_oos= \
                                self.run_trading_strategy(self.forecasts_garch_oos, self.signal_garch_oos, \
                                                        self.long_optimal_garch, self.short_optimal_garch, self.sig_mult_garch, \
                                                        "Out of Sample", self.name_garch, self.plots, verbose)

        def get_FIGARCH_data(self):

                initial_params = [1e-4, 0.35, 0.25] # omega, d, beta

                figarch_forecaster_bt = FIGARCH_Forecaster_1_d_0_MLE_delta_stds(
                        train_data=self.e_train_backtest,
                        test_data=self.e_test_backtest,
                        train_realised_variance=self.train_realised_variance_backtest.flatten(),
                        test_realised_variance=self.realised_variance_test_backtest.flatten(),
                        initial_params=initial_params,
                        H=self.H, 
                        M=self.M, 
                        scale=100,
                        percent_space=True)

                omega, d, beta, sigma2_prev = figarch_forecaster_bt.estimate_figarch()
                norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_sigma, forecasts_train, true_vols_train = figarch_forecaster_bt.forecast_figarch()

                if self.verbose:
                        print(f"\nFIGARCH backtest estimation results:")
                        print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")

                # get decimal forecasts that line up with implied vols (1: so signal is based on our forecast for today vs implied vol yday)
                realised_vol_forecasts = realised_vol_H_forecasts[1:] / 100
                self.forecasts_figarch_backtest = realised_vol_forecasts[:int(self.N_test * self.train_size)]

                stds_vol = stds_of_sigma[1:]
                stds_vol_backtest = stds_vol[:int(self.N_test * self.train_size)]
                signal_backtest = 1 / stds_vol_backtest  
                self.signal_figarch_backtest = (signal_backtest - signal_backtest.min()) / (signal_backtest.max() - signal_backtest.min())

                self.forecasts_train_ols_figarch_backtest = forecasts_train / 100
                self.true_vol_series_train_ols_figarch_backtest = true_vols_train / 100

                figarch_forecaster_oos = FIGARCH_Forecaster_1_d_0_MLE_delta_stds(
                        train_data=self.e_train_oos,
                        test_data=self.e_test_oos,
                        train_realised_variance=self.train_realised_variance_oos.flatten(),
                        test_realised_variance=self.realised_variance_test_oos.flatten(),
                        initial_params=initial_params,
                        H=self.H, 
                        M=self.M, 
                        scale=100,
                        percent_space=True)

                omega, d, beta, sigma2_prev = figarch_forecaster_oos.estimate_figarch()
                norm_mse, norm_mae, R2, Qlike, realised_vol_H_forecasts, realised_vol_H_true, stds_of_sigma, forecasts_train, _ = figarch_forecaster_oos.forecast_figarch()

                if self.verbose:
                        print(f"\n FIGARCH out of sample estimation results:")
                        print(f"Norm MSE: {norm_mse}, Norm MAE: {norm_mae}, R2: {R2}")

                # get decimal forecasts that line up with implied vols - again one forward so no forward look when comparing to implied vols
                self.forecasts_figarch_oos = realised_vol_H_forecasts[1:] / 100
                signal_oos = 1 / stds_of_sigma[1:]  

                # we want to normalise signals without adding forward look bias
                bt_min = signal_backtest.min()
                bt_max = signal_backtest.max()

                norm_signal_oos = []
                cur_min, cur_max = bt_min, bt_max

                for val in signal_oos:

                    norm_signal_oos.append((val - cur_min) / (cur_max - cur_min))

                    cur_min = min(cur_min, val)
                    cur_max = max(cur_max, val)

                norm_signal_oos = np.array(norm_signal_oos)
                self.signal_figarch_oos = norm_signal_oos 

                self.forecasts_train_ols_figarch_oos = forecasts_train / 100

        def get_FIGARCH_hyperparams_and_run_strategy(self, verbose):

                self.get_FIGARCH_data()

                self.long_optimal_figarch, self.short_optimal_figarch, self.sig_mult_figarch, self.alpha_implied_figarch = \
                        self.get_hyperparameters(self.forecasts_figarch_backtest, self.signal_figarch_backtest, self.forecasts_train_ols_figarch_backtest, \
                            self.true_vol_series_train_ols_figarch_backtest)
                
                self.name_figarch = 'FIGARCH'

                if self.OLS_INDICATOR:
                    self.forecasts_figarch_backtest, self.smear_figarch, self.beta_figarch, self.scalar_figarch = regression_forecasts_ridge_train(self.forecasts_train_ols_figarch_backtest, \
                                                                 self.atm_implied_vol_data['Mid'].values[:len(self.forecasts_train_ols_figarch_backtest)], self.alpha_implied_figarch, \
                                                                 self.true_vol_series_train_ols_figarch_backtest, self.forecasts_figarch_backtest, \
                                                                 self.atm_mid_implied_vol_backtest)
                    self.name_figarch = 'FIGARCH with Implied'
                     
                self.cash_figarch_bt, self.trading_portfolio_value_figarch_bt, self.total_portfolio_value_figarch_bt, \
                        self.trading_summary_figarch_bt, self.option_delta_positions_figarch_bt, self.delta_hedge_figarch_bt, \
                        self.delta_hedge_cashflows_figarch_bt, self.hedge_carry_figarch_bt, self.daily_option_pnl_real_figarch_bt, \
                        self.daily_option_pnl_mtm_figarch_bt,self.daily_hedge_pnl_mtm_figarch_bt,self.sharpe_figarch_bt, self.max_dd_figarch_bt, \
                        self.calmar_figarch_bt, self.cagr_figarch_bt, _, self.precision_all_figarch_bt, self.precision_active_figarch_bt, \
                        self.win_loss_ratio_all_figarch_bt, self.win_loss_ratio_active_figarch_bt, self.avg_win_all_figarch_bt, self.avg_win_active_figarch_bt, \
                        self.avg_loss_all_figarch_bt, self.avg_loss_active_figarch_bt, self.non_active_share_figarch_bt, self.total_trades_figarch_bt = \
                                self.run_trading_strategy(self.forecasts_figarch_backtest, self.signal_figarch_backtest, \
                                                        self.long_optimal_figarch, self.short_optimal_figarch, self.sig_mult_figarch, \
                                                        "Backtest", self.name_figarch, self.plots, verbose)

                if self.OLS_INDICATOR:
                    self.forecasts_figarch_oos = regression_forecasts_ridge_test(self.forecasts_figarch_oos, \
                                                                                self.atm_mid_implied_vol_oos, self.smear_figarch, \
                                                                                self.beta_figarch, self.scalar_figarch)
                     
                self.cash_figarch_oos, self.trading_portfolio_value_figarch_oos, self.total_portfolio_value_figarch_oos, \
                        self.trading_summary_figarch_oos, self.option_delta_positions_figarch_oos, self.delta_hedge_figarch_oos, \
                        self.delta_hedge_cashflows_figarch_oos, self.hedge_carry_figarch_oos, self.daily_option_pnl_real_figarch_oos, \
                        self.daily_option_pnl_mtm_figarch_oos,self.daily_hedge_pnl_mtm_figarch_oos,self.sharpe_figarch_oos, self.max_dd_figarch_oos, \
                        self.calmar_figarch_oos, self.cagr_figarch_oos, _, self.precision_all_figarch_oos, self.precision_active_figarch_oos, \
                        self.win_loss_ratio_all_figarch_oos, self.win_loss_ratio_active_figarch_oos, self.avg_win_all_figarch_oos, self.avg_win_active_figarch_oos, \
                        self.avg_loss_all_figarch_oos, self.avg_loss_active_figarch_oos, self.non_active_share_figarch_oos, self.total_trades_figarch_oos = \
                                self.run_trading_strategy(self.forecasts_figarch_oos, self.signal_figarch_oos, \
                                                        self.long_optimal_figarch, self.short_optimal_figarch, self.sig_mult_figarch, \
                                                        "Out of Sample", self.name_figarch, self.plots, verbose)

        def run_all_strategies(self, verbose):

                self.prepare_universal_series()

                self.get_BMSM_hyperparams_and_run_strategy(verbose)
                self.get_GARCH_hyperparams_and_run_strategy(verbose)
                self.get_FIGARCH_hyperparams_and_run_strategy(verbose)

        def compare_strategies(self, verbose):

                self.run_all_strategies(verbose)

                plt.figure(figsize=(8, 6))
                plt.plot(self.total_portfolio_value_bmsm_oos, label=f'{self.name_bmsm} Total Portfolio Value', color='green')
                plt.plot(self.total_portfolio_value_garch_oos, label=f'{self.name_garch} Total Portfolio Value', color='b', linestyle = ":")
                plt.plot(self.total_portfolio_value_figarch_oos, label=f'{self.name_figarch} Total Portfolio Value', color='r', linestyle='--')
                plt.xlabel("Day", fontsize=16)
                plt.ylabel("Total Portfolio Value", fontsize=16)
                plt.title("Total Portfolio Value Over Time (USD) - BMSM, GARCH and FIGARCH (Out of Sample)", fontsize=20)
                plt.legend(fontsize = 'xx-large')
                plt.grid()
                plt.tight_layout()

                plt.figure(figsize=(8, 6))
                plt.plot(self.total_portfolio_value_bmsm_bt, label=f'{self.name_bmsm} Total Portfolio Value', color='green')
                plt.plot(self.total_portfolio_value_garch_bt, label=f'{self.name_garch} Total Portfolio Value', color='b', linestyle = ":")
                plt.plot(self.total_portfolio_value_figarch_bt, label=f'{self.name_figarch} Total Portfolio Value', color='r', linestyle='--')
                plt.xlabel("Day", fontsize=16)
                plt.ylabel("Total Portfolio Value", fontsize=16)
                plt.title("Total Portfolio Value Over Time (USD) - BMSM, GARCH and FIGARCH (Backtest)", fontsize=20)
                plt.legend(fontsize = 'xx-large')
                plt.grid()
                plt.tight_layout()

                models = [
                        (self.name_bmsm, self.forecasts_bmsm_backtest, self.forecasts_bmsm_oos, self.signal_bmsm_backtest, self.signal_bmsm_oos),
                        (self.name_garch, self.forecasts_garch_backtest, self.forecasts_garch_oos, self.signal_garch_backtest, self.signal_garch_oos),
                        (self.name_figarch, self.forecasts_figarch_backtest, self.forecasts_figarch_oos, self.signal_figarch_backtest, self.signal_figarch_oos)
                        ]

                fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
                backtest_idx = len(self.forecasts_bmsm_backtest)

                for ax, (name, fc_bt, fc_oos, sig_bt, sig_oos) in zip(axs, models):
                        ax.plot(self.true_vol_series * 100, lw=1.4, ls=':',label='Actual Realised', color='b')
                        ax.plot(np.hstack((fc_bt, fc_oos)) * 100, lw=1, ls='-', label='Forecast', color='green', alpha = 0.8)
                        ax.plot(self.atm_mid_implied_vol_test * 100, lw=1, ls='--', label='ATM Mid Implied', color='red', alpha = 0.8)
                        ax.axvline(x=backtest_idx, color='gray', linestyle='--', lw=1, label = 'Backtest Period')
                        ax.set_ylabel('Volatility (%)', fontsize=14)
                        ax.grid(alpha=0.5)

                        ax2 = ax.twinx()
                        ax2.plot(np.hstack((sig_bt, sig_oos)), lw=1, ls='--', label='Signal Strength', color='orange', alpha=0.4)
                        ax2.set_ylabel('Signal Strength', fontsize=14, color='orange')

                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', edgecolor='black', framealpha=0.7, fontsize=10)

                        ax.set_title(f'{name} Forecasts vs. Realised Volatility')

                axs[-1].set_xlabel('Date', fontsize=14)
                fig.suptitle(f'{self.H}-Day Annualised Volatility: Model vs. Actual with Signal Strength for {self.ticker}', fontsize=18, y=1.02)
                fig.tight_layout()

                if self.plots:
                        plt.show()

                summary_in_domestic = {
                        f"{self.name_bmsm} Backtest": {
                                "long_threshold": self.long_optimal_bmsm,
                                "short_threshold": self.short_optimal_bmsm,
                                "signal_multiplier": self.sig_mult_bmsm,
                                "alpha_implied": self.alpha_implied_bmsm,
                                "sharpe_ratio (USD)": self.sharpe_bmsm_bt,
                                "max_drawdown (USD)": self.max_dd_bmsm_bt,
                                "calmar_ratio (USD)": self.calmar_bmsm_bt,
                                "cagr (USD)": self.cagr_bmsm_bt,
                                "Precision (all)": self.precision_all_bmsm_bt,
                                "Precision (active)": self.precision_active_bmsm_bt,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_bmsm_bt,
                                "Win/loss ratio (active)": self.win_loss_ratio_active_bmsm_bt,
                                "Avg win (all)": self.avg_win_all_bmsm_bt,
                                "Avg win (active)": self.avg_win_active_bmsm_bt,
                                "Avg loss (all)": self.avg_loss_all_bmsm_bt,
                                "Avg loss (active)": self.avg_loss_active_bmsm_bt,
                                "Non-active share": self.non_active_share_bmsm_bt,
                                "Total trades": self.total_trades_bmsm_bt
                        },

                        f"{self.name_bmsm} Out of Sample": {
                                "long_threshold": self.long_optimal_bmsm,
                                "short_threshold": self.short_optimal_bmsm,
                                "signal_multiplier": self.sig_mult_bmsm,
                                "alpha_implied": self.alpha_implied_bmsm,
                                "sharpe_ratio (USD)": self.sharpe_bmsm_oos,
                                "max_drawdown (USD)": self.max_dd_bmsm_oos,
                                "calmar_ratio (USD)": self.calmar_bmsm_oos,
                                "cagr (USD)": self.cagr_bmsm_oos,
                                "Precision (all)": self.precision_all_bmsm_oos,
                                "Precision (active)": self.precision_active_bmsm_oos,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_bmsm_oos,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_bmsm_oos,
                                "Win/loss ratio (active)": self.win_loss_ratio_active_bmsm_oos,
                                "Avg win (all)": self.avg_win_all_bmsm_oos,
                                "Avg win (active)": self.avg_win_active_bmsm_oos,
                                "Avg loss (all)": self.avg_loss_all_bmsm_oos,
                                "Avg loss (active)": self.avg_loss_active_bmsm_oos,
                                "Non-active share": self.non_active_share_bmsm_oos,
                                "Total trades": self.total_trades_bmsm_oos
                        },

                        f"{self.name_garch} Backtest": {
                                "long_threshold": self.long_optimal_garch,
                                "short_threshold": self.short_optimal_garch,
                                "signal_multiplier": self.sig_mult_garch,
                                "alpha_implied": self.alpha_implied_garch,
                                "sharpe_ratio (USD)": self.sharpe_garch_bt,
                                "max_drawdown (USD)": self.max_dd_garch_bt,
                                "calmar_ratio (USD)": self.calmar_garch_bt,
                                "cagr (USD)": self.cagr_garch_bt,
                                "Precision (all)": self.precision_all_garch_bt,
                                "Precision (active)": self.precision_active_garch_bt,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_garch_bt,
                                "Win/loss ratio (active)": self.win_loss_ratio_active_garch_bt,
                                "Avg win (all)": self.avg_win_all_garch_bt,
                                "Avg win (active)": self.avg_win_active_garch_bt,
                                "Avg loss (all)": self.avg_loss_all_garch_bt,
                                "Avg loss (active)": self.avg_loss_active_garch_bt,
                                "Non-active share": self.non_active_share_garch_bt,
                                "Total trades": self.total_trades_garch_bt
                        },

                        f"{self.name_garch} Out of Sample": {
                                "long_threshold": self.long_optimal_garch,
                                "short_threshold": self.short_optimal_garch,
                                "signal_multiplier": self.sig_mult_garch,
                                "alpha_implied": self.alpha_implied_garch,
                                "sharpe_ratio (USD)": self.sharpe_garch_oos,
                                "max_drawdown (USD)": self.max_dd_garch_oos,
                                "calmar_ratio (USD)": self.calmar_garch_oos,
                                "cagr (USD)": self.cagr_garch_oos,
                                "Precision (all)": self.precision_all_garch_oos,
                                "Precision (active)": self.precision_active_garch_oos,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_garch_oos,
                                "Win/loss ratio (active)": self.win_loss_ratio_active_garch_oos,
                                "Avg win (all)": self.avg_win_all_garch_oos,
                                "Avg win (active)": self.avg_win_active_garch_oos,
                                "Avg loss (all)": self.avg_loss_all_garch_oos,
                                "Avg loss (active)": self.avg_loss_active_garch_oos,
                                "Non-active share": self.non_active_share_garch_oos,
                                "Total trades": self.total_trades_garch_oos
                        },

                        f"{self.name_figarch} Backtest": {
                                "long_threshold": self.long_optimal_figarch,
                                "short_threshold": self.short_optimal_figarch,
                                "signal_multiplier": self.sig_mult_figarch,
                                "alpha_implied": self.alpha_implied_figarch,
                                "sharpe_ratio (USD)": self.sharpe_figarch_bt,
                                "max_drawdown (USD)": self.max_dd_figarch_bt,
                                "calmar_ratio (USD)": self.calmar_figarch_bt,
                                "cagr (USD)": self.cagr_figarch_bt,
                                "Precision (all)": self.precision_all_figarch_bt,
                                "Precision (active)": self.precision_active_figarch_bt,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_figarch_bt,
                                "Win/loss ratio (active)": self.win_loss_ratio_active_figarch_bt,
                                "Avg win (all)": self.avg_win_all_figarch_bt,
                                "Avg win (active)": self.avg_win_active_figarch_bt,
                                "Avg loss (all)": self.avg_loss_all_figarch_bt,
                                "Avg loss (active)": self.avg_loss_active_figarch_bt,
                                "Non-active share": self.non_active_share_figarch_bt,
                                "Total trades": self.total_trades_figarch_bt
                        },

                        f"{self.name_figarch} Out of Sample": {
                                "long_threshold": self.long_optimal_figarch,
                                "short_threshold": self.short_optimal_figarch,
                                "signal_multiplier": self.sig_mult_figarch,
                                "alpha_implied": self.alpha_implied_figarch,
                                "sharpe_ratio (USD)": self.sharpe_figarch_oos,
                                "max_drawdown (USD)": self.max_dd_figarch_oos,
                                "calmar_ratio (USD)": self.calmar_figarch_oos,
                                "cagr (USD)": self.cagr_figarch_oos,
                                "Precision (all)": self.precision_all_figarch_oos,
                                "Precision (active)": self.precision_active_figarch_oos,
                                "Win/loss ratio (all)": self.win_loss_ratio_all_figarch_oos,
                                "Win/loss ratio (active)": self.win_loss_ratio_active_figarch_oos,
                                "Avg win (all)": self.avg_win_all_figarch_oos,
                                "Avg win (active)": self.avg_win_active_figarch_oos,
                                "Avg loss (all)": self.avg_loss_all_figarch_oos,
                                "Avg loss (active)": self.avg_loss_active_figarch_oos,
                                "Non-active share": self.non_active_share_figarch_oos,
                                "Total trades": self.total_trades_figarch_oos
                        }}
                
                df = pd.DataFrame(summary_in_domestic).T
                df['cagr (USD)'] = df['cagr (USD)'].apply(lambda x: f"{x * 100:.2f}%")
                df['sharpe_ratio (USD)'] = df['sharpe_ratio (USD)'].apply(lambda x: f"{x:.2f}")
                df['max_drawdown (USD)'] = df['max_drawdown (USD)'].apply(lambda x: f"{x:.2f}")
                df['calmar_ratio (USD)'] = df['calmar_ratio (USD)'].apply(lambda x: f"{x:.2f}")
                df['long_threshold'] = df['long_threshold'].apply(lambda x: f"{x:.2f}")
                df['short_threshold'] = df['short_threshold'].apply(lambda x: f"{x:.2f}")
                df['signal_multiplier'] = df['signal_multiplier'].apply(lambda x: f"{x:.2f}")
                df['alpha_implied'] = df['alpha_implied'].apply(lambda x: f"{x:.2f}")
                df['Precision (all)'] = df['Precision (all)'].apply(lambda x: f"{x:.2f}")
                df['Precision (active)'] = df['Precision (active)'].apply(lambda x: f"{x:.2f}")
                df['Win/loss ratio (all)'] = df['Win/loss ratio (all)'].apply(lambda x: f"{x:.2f}")
                df['Win/loss ratio (active)'] = df['Win/loss ratio (active)'].apply(lambda x: f"{x:.2f}")
                df['Avg win (all)'] = df['Avg win (all)'].apply(lambda x: f"{x*100:.2f}%")
                df['Avg win (active)'] = df['Avg win (active)'].apply(lambda x: f"{x*100:.2f}%")
                df['Avg loss (all)'] = df['Avg loss (all)'].apply(lambda x: f"{x*100:.2f}%")
                df['Avg loss (active)'] = df['Avg loss (active)'].apply(lambda x: f"{x*100:.2f}%")
                df['Non-active share'] = df['Non-active share'].apply(lambda x: f"{x:.2f}")
                df['Total trades'] = df['Total trades'].apply(lambda x: f"{x:.2f}")

                results_in_domestic = df.style.set_properties(**{
                'color': 'white',        
                'border-color': 'black' 
                })

                return results_in_domestic
        