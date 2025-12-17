import matplotlib.pyplot as plt
import numpy as np
from torch import no_grad, tensor, cat

def plotFuturePredictions(_model, _dataset, _device, daysAhead=5):
    """
    Predict multiple days into the future (autoregressive)
    """
    _model.eval()
    
    # Get the last sequence
    lastIdx = len(_dataset) - 1
    lastDates, lastX, lastY = _dataset[lastIdx]
    
    predictions = []
    actualsIfAvailable = []
    dates = ["Today"]
    
    # Today's actual price
    todayPrice = float(_dataset.inverseTransform(np.array([lastY.item()]))[0])
    predictions.append(todayPrice)
    
    # Predict future days (autoregressive)
    currentSequence = lastX.clone()  # Start with last known sequence
    
    for day in range(1, daysAhead + 1):
        with no_grad():
            # Add batch dimension and predict
            predNormalized = _model(currentSequence.unsqueeze(0).to(_device)).cpu().item()
            
            # Denormalize
            predOriginal = float(_dataset.inverseTransform(np.array([predNormalized]))[0])
            predictions.append(predOriginal)
            dates.append(f"Day +{day}")
            
            # Update sequence for next prediction (autoregressive)
            # Remove oldest, add prediction as new "Close" price
            newSequence = currentSequence[1:].clone()
            
            # Create new data point (use same Open as previous Close, or average)
            # This is simplified - in reality you'd need more sophisticated logic
            lastPoint = newSequence[-1].clone()
            newPoint = tensor([
                lastPoint[1],  # Open = previous Close
                predNormalized,  # Close = prediction
                lastPoint[2]   # Volume = keep same (or predict)
            ])
            
            newSequence = cat([newSequence, newPoint.unsqueeze(0)])
            currentSequence = newSequence
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # Historical context (last 10 days)
    historicalPrices = []
    historicalDates = []
    for i in range(max(0, lastIdx-9), lastIdx+1):
        _, _, yVal = _dataset[i]
        price = float(_dataset.inverseTransform(np.array([yVal.item()]))[0])
        historicalPrices.append(price)
        historicalDates.append(f"D-{lastIdx-i}")
    
    # Plot historical
    plt.plot(historicalDates, historicalPrices, 'bo-', 
             label='Historical', linewidth=2, markersize=6)
    
    # Plot predictions
    predictionIndices = range(len(historicalDates)-1, len(historicalDates)+len(predictions)-1)
    allDates = historicalDates + dates[1:]  # Skip "Today" since it's in historical
    allPrices = historicalPrices + predictions[1:]
    
    plt.plot(predictionIndices, predictions, 'ro--', 
             label='Predictions', linewidth=3, markersize=10, marker='X')
    
    # Fill uncertainty (simple version - could use prediction intervals)
    plt.fill_between(predictionIndices, 
                     [p * 0.98 for p in predictions],  # -2%
                     [p * 1.02 for p in predictions],  # +2%
                     alpha=0.2, color='red', label='±2% Uncertainty')
    
    plt.xlabel('Timeline', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title(f'{daysAhead}-Day Price Forecast', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (date, price) in enumerate(zip(allDates[-len(predictions):], predictions)):
        plt.annotate(f'${price:.2f}', 
                    xy=(predictionIndices[i], price), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add summary
    totalChange = ((predictions[-1] / predictions[0]) - 1) * 100
    avgDailyChange = totalChange / daysAhead
    
    plt.text(0.02, 0.98, 
             f'Forecast Summary:\nStart: ${predictions[0]:.2f}\nEnd: ${predictions[-1]:.2f}\nTotal Change: {totalChange:+.1f}%\nAvg Daily: {avgDailyChange:+.1f}%',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return predictions