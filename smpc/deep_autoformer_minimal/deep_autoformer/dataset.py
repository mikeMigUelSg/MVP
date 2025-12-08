
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SlidingWindowDataset(Dataset):
    """
    Generic sliding-window dataset for univariate/multivariate series.
    Assumes a CSV with columns: timestamp, target, [covariates...]
    """
    def __init__(self, csv_path, seq_len=96, label_len=48, pred_len=24,
                 target_col="target", feature_cols=None, scaler=True,
                 start_date=None, end_date=None):
        df = pd.read_csv(csv_path, parse_dates=[0])
        df = df.sort_values(df.columns[0])

        # Filtrar por datas se especificado
        if start_date is not None or end_date is not None:
            date_col = df.columns[0]
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                df = df[df[date_col] >= start_date]
                print(f"  Filtered data from {start_date} onwards")
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                df = df[df[date_col] <= end_date]
                print(f"  Filtered data until {end_date}")
            print(f"  Date range after filtering: {df[date_col].min()} to {df[date_col].max()}")
            print(f"  Samples after date filtering: {len(df)}")
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in [df.columns[0], target_col]]
        self.target = df[target_col].values.astype(np.float32).reshape(-1, 1)
        self.cov = df[feature_cols].values.astype(np.float32) if feature_cols else None

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.scaler_y = StandardScaler() if scaler else None
        self.scaler_x = StandardScaler() if scaler and self.cov is not None else None

        if self.scaler_y is not None:
            self.target = self.scaler_y.fit_transform(self.target)
        if self.cov is not None and self.scaler_x is not None:
            self.cov = self.scaler_x.fit_transform(self.cov)

        # compose features
        if self.cov is not None:
            self.series = np.concatenate([self.target, self.cov], axis=1)
        else:
            self.series = self.target
        self.n = len(self.series)

    def __len__(self):
        return self.n - (self.seq_len + self.pred_len)

    def __getitem__(self, idx):
        s = idx
        e = s + self.seq_len
        l = e - self.label_len
        p = e + self.pred_len

        enc = self.series[s:e]                  # [Le, Din]
        # decoder input: last label_len targets (as known context) + zeros for future covs/targets
        dec_inp = np.zeros((self.label_len + self.pred_len, self.series.shape[1]), dtype=np.float32)
        dec_inp[:self.label_len, :1] = self.series[l:e, :1]  # only target known
        # (optional) you can pass known future covariates here if you have them

        y = self.series[e:p, :1]               # target future

        enc = torch.from_numpy(enc)
        dec_inp = torch.from_numpy(dec_inp)
        y = torch.from_numpy(y)

        return enc, dec_inp, y

    def inverse_transform_y(self, y):
        if self.scaler_y is None:
            return y
        return self.scaler_y.inverse_transform(y)
