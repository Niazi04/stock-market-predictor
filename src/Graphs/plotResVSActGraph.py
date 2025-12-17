import matplotlib.pyplot as plt
import numpy as np

def plotResidualsVsActual(testLoader, dataset, _predictions, _actuals):
    """
    Residuals vs Actual Values: Check for heteroscedasticity
    """
    # Denormalize
    predictionsOriginal = dataset.inverseTransform(_predictions)
    actualsOriginal = dataset.inverseTransform(_actuals)
    
    residuals = predictionsOriginal - actualsOriginal
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(actualsOriginal, residuals, 
                         c=np.abs(residuals),  # Color by absolute error
                         cmap='viridis', 
                         alpha=0.7, 
                         s=50,
                         edgecolors='black',
                         linewidth=0.5)
    
    plt.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Zero Error Line')
    
    # Add LOESS smooth line to detect patterns
    # try:
    #     from statsmodels.nonparametric.smoothers_lowess import lowess
    #     lowessLine = lowess(residuals, actualsOriginal, frac=0.3)
    #     plt.plot(lowessLine[:, 0], lowessLine[:, 1], 
    #             'r-', linewidth=2, alpha=0.7, label='Trend Line')
    # except:
    #     pass
    
    plt.xlabel('Actual Closing Price ($)', fontsize=12)
    plt.ylabel('Residual / Error ($)', fontsize=12)
    plt.title('Residuals vs Actual Values: Checking Homoscedasticity', 
              fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Absolute Error ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    correlation = np.corrcoef(actualsOriginal, residuals)[0, 1]
    correlationText = "No correlation" if abs(correlation) < 0.1 else \
                      "Weak correlation" if abs(correlation) < 0.3 else \
                      "Moderate correlation" if abs(correlation) < 0.5 else "Strong correlation"
    
    plt.text(0.02, 0.98, 
             f'Correlation: {correlation:.3f}\n{correlationText}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return actualsOriginal, residuals