import matplotlib.pyplot as plt
from torch import mean, no_grad

def plotFeatureImportance(_model, _dataset, _device):
    """
    Analyze which features (Open, Close, Volume) are most important
    """
    _model.eval()
    
    testSequences = []
    featureNames = ['Open', 'Close', 'Volume']
    
    # Get a sample sequence
    _, sampleX, _ = _dataset[0]
    
    for i in range(3):
        modifiedX = sampleX.clone()
        # Zero out this feature
        modifiedX[:, i] = mean(modifiedX[:, i])
        testSequences.append(modifiedX)
    
    # Get baseline prediction
    with no_grad():
        baseline = _model(sampleX.unsqueeze(0).to(_device)).cpu().item()
        
        featureImpacts = []
        for seq in testSequences:
            pred = _model(seq.unsqueeze(0).to(_device)).cpu().item()
            impact = abs(pred - baseline) / abs(baseline) * 100
            featureImpacts.append(impact)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(featureNames, featureImpacts, 
                   color=['skyblue', 'lightgreen', 'salmon'],
                   edgecolor='black', linewidth=2)
    
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Impact on Prediction (%)', fontsize=12)
    plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, impact in zip(bars, featureImpacts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{impact:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return dict(zip(featureNames, featureImpacts))