from torch import no_grad
import numpy as np

def plotHelper(_model, _device, _testLoader, _numSamples = 100):
    _model.eval()
    predictions = []
    actuals = []
    
    with no_grad():
        for i, (_, x, y) in enumerate(_testLoader):
            if len(predictions) >= _numSamples:
                break
                
            x = x.to(_device)
            y = y.to(_device)
            
            pred = _model(x)
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
    
    predictions = np.array(predictions[:_numSamples])
    actuals = np.array(actuals[:_numSamples])
    return predictions, actuals