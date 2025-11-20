import torch
import yfinance as f
import pandas as pd
from torch.utils.data import Dataset



class StocksDataset(Dataset):
    def __init__(self, sTickerSymbol: str = None, sPath: str = None, iSequenceLength: int = 60):

        df          = None
        self.sl     = iSequenceLength

        if sTickerSymbol:
            t = f.Ticker(sTickerSymbol)
            df = t.history(start="2000-01-03", end="2025-11-14")
            df = df.reset_index()
            del df['Dividends']
            del df['Stock Splits']
            # df['Date'] = df['Date'].str.split(' ').str[0]
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        
        elif sPath:
            from os.path import exists

            if not exists(sPath):
                    raise FileNotFoundError(f"No such file << {sPath} >> exists - download fallback was turned off")
            
            df = pd.read_csv(sPath)
            if 'Dividends'    in df:  del df['Dividends']
            if 'Stock Splits' in df:  del df['Stock Splits']

            df['Date'] = df['Date'].str.split(' ').str[0]

        self.data = df[["Open", "Close", "Volume"]].values.astype(float)
        self.date = df['Date'].tolist()
        

    def __len__(self):
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