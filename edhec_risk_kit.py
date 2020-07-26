import pandas as pd, numpy as np

###############################
#####      LOAD DATA     ######
###############################

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by Market Cap
    """

    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0, index_col=0, parse_dates=True, na_values=-99.99)
    me_m.index = pd.to_datetime(me_m.index, format="%Y%m")
    me_m.index = me_m.index.to_period('M')
    rets = me_m[["Lo 20", "Hi 20"]]
    rets.columns = ["SmallCap", "LargeCap"]
    rets = rets/100
    
    return rets

def get_hfi_returns():
    """
    Load EDHEC Hedge Fund Indices data
    """
    ret = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    ret = ret / 100
    ret.index = ret.index.to_period("M")
    return ret

def get_ind_returns():
    """
    Get industry returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_ind_nfirms():
    """
    Get industry number of firms
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_ind_size():
    """
    Get industry size 
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_total_market_index_returns():
    """
    Get and calculate the total market index returns
    """
    ind_ret = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis="columns")
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    ind_capweight["1926"].sum(axis="columns")
    
    total_mkt_ret = (ind_capweight*ind_ret).sum(axis="columns")
    
    return total_mkt_ret
    

###############################
###     CALCULATIONS     ######
###############################

def annualized_ret(ret, periods_per_year=12):
    return (ret+1).prod()**(periods_per_year/ret.shape[0])-1

def annualized_vol(ret, periods_per_year=12):
    return ret.std() * (periods_per_year**0.5)

def sharpe_ratio(ret, risk_free_rate = 0, periods_per_year=12):
    return (annualized_ret(ret, periods_per_year) - risk_free_rate) / annualized_vol(ret, periods_per_year)

def drawdown(ret_series : pd.Series):
    """
    Calculate Drawdowns
    """
    wealth = (ret_series + 1).cumprod()
    previous_peaks = wealth.cummax()
    drawdown = (wealth - previous_peaks)/previous_peaks
    drawdown
    return pd.DataFrame({
        "Wealth": wealth,
        "Previous Peak": previous_peaks,
        "Drawdown": drawdown
    })

def semi_std(ret):
    """
    Calculate semi standard deviations by only considering negative returns
    """
    return ret[ret < 0].std(ddof=0)

def var_historic(ret, level=5):
    """
    VaR Historic
    """
    return -ret.quantile(level/100)

from scipy.stats import norm

def var_gaussian(r, level=5, modified=False):
    """
    Returns Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # Assume Gaussian z score
    z = norm.ppf(level/100) 
    
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = r.skew()
        k = r.kurtosis()
        z = (z +
                (z**2 - 1)*s/6 + 
                (z**3 - 3*z)*k/24 - 
                (2*z**3 - 5*z)*(s**2)/36
            )

    return -(r.mean() + z * r.std(ddof=0))

def cvar_historic(r, level=5, modified=False):
    """
    CVar historic
    """
    return r[r < r.quantile(level/100)].mean()

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame with aggregated summary of returns
    """
    return pd.DataFrame({
        "Annualized Return": r.aggregate(annualized_ret, periods_per_year=12),
        "Annualized Vol": r.aggregate(annualized_vol, periods_per_year=12),
        "Skewness": r.skew(),
        "Kurtosis": r.kurtosis(),
        "Cornish-Fisher VaR (5%)": r.aggregate(var_gaussian, modified=True),
        "Historic CVaR": r.aggregate(cvar_historic),
        "Max Drawdown": r.aggregate(lambda ret: drawdown(ret).Drawdown.min())
    })

################################
###### Efficient Frontier ######
################################

def portfolio_ret(weights, returns):
    """
    Calculate returns of portfolio from weights
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Calculates Vol from weight
    """
    return (weights.T @ covmat @ weights)**0.5

from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0, 1), ) *n
    return_is_target = {
        'type':'eq',
        'args':(er,),
        'fun': lambda weights, er: target_return - portfolio_ret(weights, er)
    }
    sum_to_one = {
        'type':'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    }
    results = minimize(portfolio_vol, init_guess, 
                      args=(cov,), method="SLSQP",
                      options={'disp': False},
                      constraints=(return_is_target, sum_to_one),
                      bounds=bounds)
    return results.x

def optimal_weights(n_points, er, cov):
    """
    Calculate list of weights to run on the optimizer on to minimize the vol
    """
    
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(r, er, cov) for r in target_rs]
    return weights

def msr(riskfree_rate, er, cov):
    """
    RiskFree rate + ER + Cov -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0, 1), ) *n
    sum_to_one = {
        'type':'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        return -(portfolio_ret(weights, er) - riskfree_rate) / (portfolio_vol(weights, cov))
    
    results = minimize(neg_sharpe_ratio, init_guess, 
                      args=(riskfree_rate, er, cov,), method="SLSQP",
                      options={'disp': False},
                      constraints=(sum_to_one),
                      bounds=bounds)
    return results.x

def gmv(cov):
    """
    Return weights for Global Minimum Variance portfolio by giving the same return for all assets
    """
    
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def plot_ef(n_points, er, cov, show_cml=False, style=".-", riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots general efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_ret(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols,
    })
    
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_ret(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], marker="o", markersize=10, color="goldenrod")
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_ret(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], marker="o", markersize=10, color="blue")
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_ret(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = (0, vol_msr)
        cml_y = (riskfree_rate, r_msr)
        ax.plot(cml_x, cml_y, color="green", marker="o", markersize=10, linestyle="dashed")
    
    return ax

##################################
######       CPPI.       #########
##################################
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy
    Returns a dictionary of: Asset Value history, Risk budget history, risky weight history
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12

    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = np.maximum(np.minimum(m * cushion, 1),0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        ## Update the account value for this time step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])

        # save the values to look up history and plot
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        
    risky_wealth = start * (1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    
    return backtest_result

################################
########   GBM.   ##############
################################

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, prices=True, steps_per_year=12, s_0=100.0):
    """
    Evolution of stockprice using Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    rets_plus_1 = np.random.normal(loc=1+mu*dt, scale=(sigma*np.sqrt(dt)), size=(n_steps+1, n_scenarios))
    rets_plus_1[0] = 1
    if prices:
        return s_0*pd.DataFrame(rets_plus_1).cumprod()
    
    return rets_plus_1-1


#######################################
######     Cox Ingersoil Ross.   ######
#######################################

def discount(t, r):
    """
    Compute the price of $1 bond given interest rate
    r can be a float, Series or DataFrame
    Returns a DataFrame indexed by t
    """
    discounts = [(1+r)**(-i) for i in t]
    return pd.DataFrame(data=discounts, index=t)

def pv(flows, r):
    """
    Computes the present value of a sequence of cashflows
    flows is indexed by time and r can be a scale, Series or DataFrame with rows matching number of rows in flows.
    Returns the PV of the sequence
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets, r)/pv(liabilities, r)

def inst_to_ann(r):
    """
    Converts short rate to an annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts annualized to short rate
    """
    return np.log1p(r)

def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,steps_per_year=12, r_0=None):
    """
    Implements the CIR model for interest rates
    """ 
    if r_0 is None: r_0 = b
    
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year)+1
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## For price generation
    h = np.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    #####
    
    def price(ttm, r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1))/(2*h + (h+a)*(np.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    #####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        prices[step] = price(n_years-step*dt, rates[step])
        
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    
    return rates, prices

def show_cir_prices(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,steps_per_year=12, r_0=0.03):
    return cir(n_years, n_scenarios, a, b, sigma,steps_per_year, r_0)[1].plot(figsize=(12, 6), legend=False)

def show_cir_rates(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,steps_per_year=12, r_0=0.03):
    return cir(n_years, n_scenarios, a, b, sigma,steps_per_year, r_0)[0].plot(figsize=(12, 6), legend=False)

############################
##### GHP liability matching
##################################

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns a series of cash flows generated by a 
    bond, indexed by a coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Price a bond that pays regular coupons until maturity
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date and bond value is computed over time
    index of discount_rate assumed to be coupon number
    Code is not efficient, but to illustrate principle
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else:  # Base case: Single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)    
        

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of sequence of cash flows, only works for 1 period
    """
    discounted_flows = discount(flows.index, discount_rate) * pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.values.flatten())

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)

    return (d_l - d_t)/(d_l - d_s)

def match_durations2(d_t, d_1, d_2):
    return (d_2 - d_t)/(d_2 - d_1)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a bond based on monthly bond prices and coupon payments
    Assumes that dividends are paid out at the end of the period and dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


#######################################
#####.   BACKTESTING.       ###########
#######################################

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a backtest (simulation) of allocating between two sets of ret
    r1 and r2 are T x N DataFrames or returns where T is the time step index and
    N is number of scenarios
    allocator is a function that takes two sets of rets and allocator specific parameters,
    and produces and allocation to the first portfolio as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that dont match r1 shape")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames, each column is a scenario, each row is the price for a timestep
    Returns and T x N DataFrame of PSP weights
    """    
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    n_steps = r1.shape[0]
    n_scenarios = r1.shape[1]
    path = pd.Series(np.linspace(start_glide, end_glide, n_steps))
    paths = pd.concat([path]*n_scenarios, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without violating the floor
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing weights in PSP
    """
    if psp_r.shape != zc_prices.shape: 
        raise ValueError("PSP and ZC prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of floor assuming today's rates and flat yc
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1)  # Same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # Recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without violating the floor
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing weights in PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1)  # Same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # Recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history 

def terminal_values(rets):
    """
    Returns the final values at the end of the return period for each scenario
    """
    return (rets+1).prod()

def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Produce summary stats on the terminal values per invested dollar
    across a range of N scenarios
    rets is a Tx N DataFrame of returns, where T is the time-step
    Returns is a 1 column DataFrame of summary stats indexed by the stat name
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (terminal_wealth[reach]-cap).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    
    return sum_stats
    