import matplotlib.pyplot as plt
from torch import no_grad
import numpy as np

def predictTestSet(model, testLoader, device, dataset):
    model.eval()
    model.to(device)
    
    allPredictions = []
    allActualsNormalized = []
    allDates = []
    
    with no_grad():
        for batchDates, xBatch, yBatch in testLoader:
            xBatch = xBatch.to(device)
            yBatch = yBatch.to(device)
            
            predictions = model(xBatch)
            
            allPredictions.extend(predictions.cpu().numpy().flatten())
            allActualsNormalized.extend(yBatch.cpu().numpy().flatten())
            
            # Store dates
            for dateList in batchDates:
                allDates.append(dateList[-1])  # Last date in sequence
    
    # Convert to numpy
    predictionsNormalized = np.array(allPredictions)
    actualsNormalized = np.array(allActualsNormalized)
    
    # Denormalize using dataset's inverseTransform
    predictionsOriginal = dataset.inverseTransform(predictionsNormalized)
    actualsOriginal = dataset.inverseTransform(actualsNormalized)
    
    return {
        'dates': allDates,
        'predictionsNormalized': predictionsNormalized,
        'actualsNormalized': actualsNormalized,
        'predictionsOriginal': predictionsOriginal,
        'actualsOriginal': actualsOriginal
    }

def plotPredictionsVsActual(resultsDict, numSamples=50):
    # Take first N samples for cleaner plot
    n = min(numSamples, len(resultsDict['predictionsOriginal']))
    
    predictions = resultsDict['predictionsOriginal'][:n]
    actuals = resultsDict['actualsOriginal'][:n]
    dates = resultsDict['dates'][:n]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Bar chart comparison
    xPos = np.arange(n)
    ax1.bar(xPos - 0.2, actuals, width=0.4, label='Actual', color='blue', alpha=0.7)
    ax1.bar(xPos + 0.2, predictions, width=0.4, label='Predicted', color='red', alpha=0.7)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'Actual vs Predicted Closing Prices (First {n} Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on last few bars only (to avoid clutter)
    for i in range(max(0, n-5), n):
        ax1.text(i - 0.2, actuals[i], f'${actuals[i]:.2f}', 
                ha='center', va='bottom', fontsize=8, rotation=45)
        ax1.text(i + 0.2, predictions[i], f'${predictions[i]:.2f}', 
                ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Plot 2: Time series with error
    ax2.plot(actuals, label='Actual', marker='o', linewidth=2, color='blue')
    ax2.plot(predictions, label='Predicted', marker='x', linestyle='--', linewidth=2, color='red')
    
    # Fill error area
    ax2.fill_between(range(n), predictions, actuals, alpha=0.2, color='gray')
    
    ax2.set_xlabel('Days into Future')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Prediction Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    # Add metrics to plot
    metricsText = f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}\nMAPE: {mape:.1f}%'
    fig.text(0.02, 0.98, metricsText, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape}