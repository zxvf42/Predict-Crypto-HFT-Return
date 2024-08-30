from utils.numba_utils import *
from utils.pandas_utils import *
from tqdm import tqdm
@pjit
def rolling_apply(ticker, trades, func, ticker_start_idx, ticker_end_idx, trades_start_idx, trades_end_idx):
    assert len(ticker_start_idx) == len(ticker_end_idx) == len(trades_start_idx) == len(trades_end_idx)
    n = len(ticker_start_idx)
    res = np.full(n, np.nan)
    for i in nb.prange(n):
        trades_window = trades[trades_start_idx[i]:trades_end_idx[i]]
        ticker_window = ticker[ticker_start_idx[i]:ticker_end_idx[i]]
        if len(trades_window) >= 2:
            res[i] = func(ticker=ticker_window, trades=trades_window)
    return res

def backward_apply(ticker, trades, clock, delta1, delta2, funcs, names):
    assert clock in ['trading', 'natural']
    assert len(funcs) == len(names)
    assert delta1 < delta2
    if clock == 'natural':
        # Int: (T-delta2, T-delta1]
        interval_start_ts = ticker['ts'] - delta2
        interval_end_ts = ticker['ts'] - delta1
        ticker_start_idx = np.searchsorted(ticker['ts'], interval_start_ts, side='right')
        ticker_end_idx = np.searchsorted(ticker['ts'], interval_end_ts, side='right')
        trades_start_idx = np.searchsorted(trades['ts'], interval_start_ts, side='right')
        trades_end_idx = np.searchsorted(trades['ts'], interval_end_ts, side='right')
    else:
        # Int: (T-delta2, T-delta1]
        # for each ticker ts at time T, find the nearest trade in (T, +infin]
        nearest_future_trade_ts_idx = np.searchsorted(trades["ts"], ticker["ts"], side="right")
        # the idx of ts right before n=delta1 trades occur is nearest_future_trade_ts_idx - delta1, 
        # and right after n=delta2 trades occur is nearest_future_trade_ts_idx - delta2
        # therefore for trades, the start idx and end idx are
        trades_end_idx = np.clip(nearest_future_trade_ts_idx - delta1, 0, len(trades))
        trades_start_idx = np.clip(nearest_future_trade_ts_idx - delta2, 0, len(trades))
        # for ticker, the start ts is
        ticker_start_ts = trades["ts"][np.clip(trades_start_idx-1, 0, len(trades))]
        # and corresponding idx in ticker is
        ticker_start_idx = np.searchsorted(ticker["ts"], ticker_start_ts, side="right")
        # the end ts is
        ticker_end_ts = trades["ts"][np.clip(trades_end_idx - 1, 0, len(trades))]
        # and corresponding idx in ticker is
        ticker_end_idx = np.searchsorted(ticker["ts"], ticker_end_ts, side="right")
        
    res = {}
    for i in tqdm(range(len(funcs))):
        func = funcs[i]
        name = names[i]
        res[name] = rolling_apply(ticker, trades, func, ticker_start_idx, ticker_end_idx, trades_start_idx, trades_end_idx)
        
    return pd.DataFrame(res)

def forward_apply(ticker, trades, clock, delta, func, name):
    assert clock in ['trading', 'natural']
    if clock == 'natural':
        interval_start_ts = ticker['ts']
        interval_end_ts = ticker['ts'] + delta
        ticker_start_idx = np.arange(len(ticker)) + 1  # np.searchsorted(ticker['ts'], interval_start_ts, side='right')
        ticker_end_idx = np.searchsorted(ticker['ts'], interval_end_ts, side='right')
        trades_start_idx = np.searchsorted(trades['ts'], interval_start_ts, side='right')
        trades_end_idx = np.searchsorted(trades['ts'], interval_end_ts, side='right')
    else:
        # Int: (T, T+delta]
        # for each ticker ts at time T, find the nearest trade in (T, +infin)
        nearest_future_trade_ts_idx = np.searchsorted(trades['ts'], ticker['ts'], side='right')
        # the idx of ts right after n=delta trades occur is nearest_future_trade_ts_idx + delta
        nearest_future_trade_ts_idx + delta
        # therefore for trades, the start idx and end idx are
        trades_start_idx = np.clip(nearest_future_trade_ts_idx, 0, len(trades))
        trades_end_idx = np.clip(nearest_future_trade_ts_idx + delta, 0, len(trades))
        # for ticker, the start idx is always
        ticker_start_idx = np.arange(len(ticker)) + 1
        # the end ts is at trades['ts'][trades_end_idx - 1]
        ticker_end_ts = trades['ts'][np.clip(trades_end_idx - 1, 0, len(trades))]
        # the corresponding idx in ticker is
        ticker_end_idx = np.searchsorted(ticker['ts'], ticker_end_ts, side='right')
        
    res = rolling_apply(ticker, trades, func, ticker_start_idx, ticker_end_idx, trades_start_idx, trades_end_idx)
    return pd.Series(res, name=name)