import matplotlib.pyplot as plt
from torch import no_grad
import numpy as np

def predictTestSet(_model, _testLoader, _device, _dataset):
    _model.eval()
    _model.to(_device)
    
    allPredictions = []
    allActualsNormalized = []
    allDates = []
    
    with no_grad():
        for batchDates, xBatch, yBatch in _testLoader:
            xBatch = xBatch.to(_device)
            yBatch = yBatch.to(_device)
            
            predictions = _model(xBatch)
            
            allPredictions.extend(predictions.cpu().numpy().flatten())
            allActualsNormalized.extend(yBatch.cpu().numpy().flatten())
            
            for dateList in batchDates:
                allDates.append(dateList[-1])  # Last date in sequence
    
    predictionsNormalized = np.array(allPredictions)
    actualsNormalized = np.array(allActualsNormalized)
    
    # Denormalize
    predictionsOriginal = _dataset.inverseTransform(predictionsNormalized)
    actualsOriginal = _dataset.inverseTransform(actualsNormalized)
    
    return {
        'dates': allDates,
        'predictionsNormalized': predictionsNormalized,
        'actualsNormalized': actualsNormalized,
        'predictionsOriginal': predictionsOriginal,
        'actualsOriginal': actualsOriginal
    }

def plotPredictionsVsActual(resultsDict, numSamples=50):
    n = min(numSamples, len(resultsDict['predictionsOriginal']))
    
    predictions = resultsDict['predictionsOriginal'][:n]
    actuals = resultsDict['actualsOriginal'][:n]
    dates = resultsDict['dates'][:n]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    xPos = np.arange(n)
    ax1.bar(xPos - 0.2, actuals, width=0.4, label='Actual', color='blue', alpha=0.7)
    ax1.bar(xPos + 0.2, predictions, width=0.4, label='Predicted', color='red', alpha=0.7)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'Actual vs Predicted Closing Prices (First {n} Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i in range(max(0, n-5), n):
        ax1.text(i - 0.2, actuals[i], f'${actuals[i]:.2f}', 
                ha='center', va='bottom', fontsize=8, rotation=45)
        ax1.text(i + 0.2, predictions[i], f'${predictions[i]:.2f}', 
                ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax2.plot(actuals, label='Actual', marker='o', linewidth=2, color='blue')
    ax2.plot(predictions, label='Predicted', marker='x', linestyle='--', linewidth=2, color='red')
    
    ax2.fill_between(range(n), predictions, actuals, alpha=0.2, color='gray')
    
    ax2.set_xlabel('Days into Future')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Prediction Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    metricsText = f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}\nMAPE: {mape:.1f}%'
    fig.text(0.02, 0.98, metricsText, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape}