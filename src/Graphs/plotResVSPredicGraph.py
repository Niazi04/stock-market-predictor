import matplotlib.pyplot as plt
import numpy as np

def plotResidualsVsPredicted(_dataset, _predictions, _actuals):
    """
    Residuals vs Predicted Values: Check for bias
    """
    
    # Denormalize
    predictionsOriginal = _dataset.inverseTransform(_predictions)
    actualsOriginal = _dataset.inverseTransform(_actuals)
    
    residuals = predictionsOriginal - actualsOriginal
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(predictionsOriginal, residuals, 
                         c=range(len(residuals)), 
                         cmap='plasma', 
                         alpha=0.7, 
                         s=50,
                         edgecolors='black',
                         linewidth=0.5)
    
    plt.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Zero Error Line')
    
    stdResiduals = np.std(residuals)
    plt.axhline(y=stdResiduals, color='orange', linestyle='--', alpha=0.5, label=f'+1 Std Dev (${stdResiduals:.2f})')
    plt.axhline(y=-stdResiduals, color='orange', linestyle='--', alpha=0.5, label=f'-1 Std Dev (${stdResiduals:.2f})')
    
    plt.xlabel('Predicted Closing Price ($)', fontsize=12)
    plt.ylabel('Residual / Error ($)', fontsize=12)
    plt.title('Residuals vs Predicted Values: Checking for Bias', 
              fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Time Order (Earlier → Later)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    meanResidual = np.mean(residuals)
    biasPercent = (meanResidual / np.mean(actualsOriginal)) * 100
    
    biasText = "No significant bias" if abs(biasPercent) < 1 else \
               "Slight bias" if abs(biasPercent) < 3 else \
               "Moderate bias" if abs(biasPercent) < 5 else "Significant bias"
    
    biasDirection = "overestimates" if meanResidual > 0 else "underestimates"
    
    plt.text(0.02, 0.98, 
             f'Mean Error: ${meanResidual:.2f}\nBias: {biasPercent:.1f}%\nModel {biasDirection} prices\n{biasText}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return predictionsOriginal, residuals