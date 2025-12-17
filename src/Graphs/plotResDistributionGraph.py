import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plotResidualDistribution(testLoader, dataset,_predictions, _actuals):
    """
    Distribution of Residuals: Histogram with normal curve
    """
    # Denormalize
    predictionsOriginal = dataset.inverseTransform(_predictions)
    actualsOriginal = dataset.inverseTransform(_actuals)
    
    residuals = predictionsOriginal - actualsOriginal
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Histogram
    n, bins, patches = plt.hist(residuals, bins=20, 
                                edgecolor='black', 
                                alpha=0.7, 
                                color='skyblue',
                                density=True,  # Show as probability density
                                label='Residuals')
    
    # Add normal distribution curve
    from scipy.stats import norm
    x = np.linspace(min(residuals), max(residuals), 100)
    meanResidual = np.mean(residuals)
    stdResidual = np.std(residuals)
    plt.plot(x, norm.pdf(x, meanResidual, stdResidual), 
             'r-', linewidth=3, alpha=0.8, 
             label=f'Normal Distribution\n(μ=${meanResidual:.2f}, σ=${stdResidual:.2f})')
    
    # Add vertical lines
    plt.axvline(x=0, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Zero Error')
    plt.axvline(x=meanResidual, color='green', linestyle='--', linewidth=2, 
                alpha=0.8, label=f'Mean (${meanResidual:.2f})')
    
    # Add ±1, ±2 standard deviation lines
    plt.axvline(x=meanResidual + stdResidual, color='orange', linestyle=':', 
                linewidth=1.5, alpha=0.5, label=f'±1σ (${stdResidual:.2f})')
    plt.axvline(x=meanResidual - stdResidual, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    if stdResidual * 2 < max(abs(residuals)):
        plt.axvline(x=meanResidual + 2*stdResidual, color='purple', linestyle=':', 
                    linewidth=1, alpha=0.3, label='±2σ')
        plt.axvline(x=meanResidual - 2*stdResidual, color='purple', linestyle=':', linewidth=1, alpha=0.3)
    
    plt.xlabel('Residual / Error ($)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Calculate statistics
    skewness = float(stats.skew(residuals))
    kurtosis = float(stats.kurtosis(residuals))
    
    statsText = f"""Error Statistics:
Mean = ${meanResidual:.2f}
Std Dev = ${stdResidual:.2f}
Skewness = {skewness:.3f}
Kurtosis = {kurtosis:.3f}
Range = [${np.min(residuals):.2f}, ${np.max(residuals):.2f}]
68% within ±${stdResidual:.2f}
95% within ±${2*stdResidual:.2f}"""
    
    plt.text(0.02, 0.98, statsText,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return residuals