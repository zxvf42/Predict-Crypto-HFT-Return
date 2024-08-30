from utils.numba_utils import *

@jit
def target_mean_txn_p(ticker, trades):
    return mean(trades['p'])

@jit
def target_last_txn_p(ticker, trades):
    return trades[-1]['p']

@jit
def target_last_mp(ticker, trades):
    return ticker[-1]['mp']