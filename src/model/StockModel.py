import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, nLayers=2):
        super(StockPredictor, self).__init__()

        self.rnn = nn.RNN(
                            input_size  = inputSize,
                            hidden_size = hiddenSize,
                            num_layers  = nLayers,
                            batch_first = True
                          )
        self.fc = nn.Linear(hiddenSize, outputSize)
    
    def forward(self, x):
        rnnOut, hn = self.rnn(x)
        last = rnnOut[:,-1,:]
        output = self.fc(last)

        return output