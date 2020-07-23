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
        "Peaks": previous_peaks,
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
