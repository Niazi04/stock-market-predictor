import torch
import yfinance as f
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class StocksDataset(Dataset):
    def __init__(self, sTickerSymbol: str = None, sPath: str = None, iSequenceLength: int = 60, bNormalize = True):
        df                  = None
        self.sl             = iSequenceLength
        self.scaler         = None
        self.data = None
        self.normalize      = bNormalize

        if sTickerSymbol:
            t = f.Ticker(sTickerSymbol)
            df = t.history(start="2000-01-03", end="2025-11-14")
            df = df.reset_index()
            del df['Dividends']
            del df['Stock Splits']
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        
        elif sPath:
            from os.path import exists

            if not exists(sPath):
                    raise FileNotFoundError(f"No such file << {sPath} >> exists")
            
            df = pd.read_csv(sPath)
            if 'Dividends'    in df:  del df['Dividends']
            if 'Stock Splits' in df:  del df['Stock Splits']

            df['Date'] = df['Date'].str.split(' ').str[0]

        rawData = df[["Open", "Close", "Volume"]].values.astype(float)

        if self.normalize:
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(rawData)
        else:
             self.data = rawData


        # self.data = df[["Open", "Close", "Volume"]].values.astype(float)
        self.date = df['Date'].tolist()
        
    def __len__(self):
        # return len(self.data) - self.sl
        return len(self.data) - self.sl
    def __getitem__(self, IDX):
        '''
            Shapes the data into:

                    ---> d: [sequence_length]
                    ---> x: (sequence_length, 3)
                    ---> y: (1)
        '''

        if IDX < 0: raise IndexError(f"StocksDataset.__getitem__() -> index={IDX} out of range")

        d = self.date[IDX: IDX + self.sl]
        x = torch.tensor(self.data[IDX: IDX + self.sl],    dtype=torch.float32) #To train over a sequence
        y = torch.tensor(self.data[IDX + self.sl][1],      dtype=torch.float32) #To predict the next day's market
        return d, x, y
    
    def inverseTransform(self, _normData):
        if not self.normalize:
            return self.data
        
        if isinstance(_normData, (int, float)):
            _normData = np.array([_normData])
        elif isinstance(_normData, torch.Tensor):
            _normData = _normData.cpu().numpy()
        
        # If it's a single value (like prediction), reshape it
        if _normData.ndim == 0:
            _normData = _normData.reshape(1, 1)
        elif _normData.ndim == 1:
            _normData = _normData.reshape(-1, 1)
        
        # Need to reconstruct full feature array for inverse transform
        # For close prices only, we need to create dummy columns
        dummyArray = np.zeros((len(_normData), 3))
        dummyArray[:, 1] = _normData.flatten()
        
        origScale = self.scaler.inverse_transform(dummyArray)        
        return origScale[:, 1] #just the closed price