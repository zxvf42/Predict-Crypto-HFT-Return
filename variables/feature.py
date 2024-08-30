from utils.numba_utils import *

# names: Breadth, Immediacy, VolumeAll, VolumeAvg, VolumeMax,
# Lambda, LobImbalance, TxnImbalance, PastReturn,
# AutoCov, QuotedSpread, EffectiveSpread,


@jit
def feature_Breadth(ticker, trades):
    return len(trades)


@jit
def feature_Immediacy(ticker, trades):
    return (trades[-1]["ts"] - trades[0]["ts"]) / len(trades)


@jit
def feature_VolumeAll(ticker, trades):
    return sum(abs(trades["v"]))


@jit
def feature_VolumeAvg(ticker, trades):
    return mean(abs(trades["v"]))


@jit
def feature_VolumeMax(ticker, trades):
    return max(abs(trades["v"]))


@jit
def feature_Lambda(ticker, trades):
    return (trades[-1]["p"] - trades[0]["p"]) / sum(abs(trades["v"]))


@jit
def feature_LobImbalance(ticker, trades):
    return mean((ticker["av"] - ticker["bv"]) / (ticker["av"] + ticker["bv"]))


@jit
def feature_TxnImbalance(ticker, trades):
    return sum(trades["v"]) / sum(abs(trades["v"]))


@jit
def feature_PastReturn(ticker, trades):
    return 1 - mean(trades["p"]) / trades[-1]["p"]


@jit
def feature_AutoCov(ticker, trades):
    log_rtn = diff(log(trades["p"]))
    return mean(log_rtn[1:] * log_rtn[:-1])


@jit
def feature_QuotedSpread(ticker, trades):
    return mean((ticker["ap"] - ticker["bp"]) / (ticker["ap"] + ticker["bp"]) * 2)


@jit
def feature_EffectiveSpread(ticker, trades):
    nominator = sum((log(trades["p"]) - log(trades["mp"])) * trades["v"] * trades["p"])
    denominator = sum(trades["p"] * abs(trades["v"]))
    return nominator / denominator
