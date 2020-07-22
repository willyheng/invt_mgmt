import pandas as pd

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

###############################
###     CALCULATIONS     ######
###############################

def annualized_ret(ret):
    if ret.index.freq == "M":
        return (ret+1).prod()**(12/ret.shape[0])-1
    else:
        raise Exception("Unrecognized frequency: ", ret.index.freq)

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
    return ret[ret < 0].std()