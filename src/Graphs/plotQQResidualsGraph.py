import matplotlib.pylab as plt
import numpy as np

def plotQQResiduals(testLoader, dataset, _predictions, _actuals):
    """
    Q-Q Plot: Check if residuals are normally distributed
    """
    # Denormalize
    predictionsOriginal = dataset.inverseTransform(_predictions)
    actualsOriginal = dataset.inverseTransform(_actuals)
    
    residuals = predictionsOriginal - actualsOriginal
    
    # Create Q-Q plot
    plt.figure(figsize=(10, 8))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    
    plt.title('Q-Q Plot: Checking Normality of Residuals', fontsize=14, fontweight='bold')
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    meanResidual = np.mean(residuals)
    stdResidual = np.std(residuals)
    
    plt.text(0.02, 0.98, 
             f'N = {len(residuals)} samples\nMean = ${meanResidual:.2f}\nStd Dev = ${stdResidual:.2f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Check normality with Shapiro-Wilk test
    from scipy.stats import shapiro
    if len(residuals) <= 5000:  # Shapiro-Wilk works for n ≤ 5000
        stat, p_value = shapiro(residuals)
        normality = "Normal" if p_value > 0.05 else "Not Normal"
        plt.text(0.02, 0.78, 
                 f'Shapiro-Wilk Test:\np-value = {p_value:.4f}\nResult: {normality}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return residuals