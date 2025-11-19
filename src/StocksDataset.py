import torch
import yfinance as f
import pandas as pd
from torch.utils.data import dataset


class StocksDataset:
    def __init__(self, sTickerSymbol=None, sPath=None, bDownloadFallBack=False):

        df = None

        if sTickerSymbol:
            t = f.Ticker(sTickerSymbol)
            df = t.history(start="2000-01-03", end="2025-11-14")
            df = df.reset_index()
            del df['Dividends']
            del df['Stock Splits']
            df['Date'] = df['Date'].str.split(' ').str[0]
        
        elif sPath:
            from os.path import exists

            if not exists(sPath):
                print(f"No such file {sPath}")
                return
            df = pd.read_csv(sPath)
            if 'Dividends'    in df:  del df['Dividends']
            if 'Stock Splits' in df:  del df['Stock Splits']

            df['Date'] = df['Date'].str.split(' ').str[0]

        
        print(df)

    def __len__(self):
        pass
    def __getitem__(self, iIDX):
        pass


StocksDataset(sPath="data.csv")