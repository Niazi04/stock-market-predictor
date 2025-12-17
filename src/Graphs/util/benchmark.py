from torch import nn, optim, no_grad

def benchmark(_model, _device, _trainLoader, _testLoader, _epochSize: int, _lr=0.001,_lossFN = nn.SmoothL1Loss()):
    _model.to(_device)
    optimizer = optim.Adam(_model.parameters(), lr=_lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, _epochSize + 1):
        # Training phase
        _model.train()
        epoch_train_loss = 0
        samples = 0
        
        for d, x, y in _trainLoader:
            optimizer.zero_grad()
            x = x.to(_device)
            y = y.to(_device).unsqueeze(1)
            
            prediction = _model(x)
            loss = _lossFN(prediction, y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * y.size(0)
            samples += y.size(0)
        
        avg_train_loss = epoch_train_loss / samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        _model.eval()
        epoch_val_loss = 0
        samples = 0
        
        with no_grad():
            for d, x, y in _testLoader:
                x = x.to(_device)
                y = y.to(_device).unsqueeze(1)
                
                prediction = _model(x)
                loss = _lossFN(prediction, y)
                
                epoch_val_loss += loss.item() * y.size(0)
                samples += y.size(0)
        
        avg_val_loss = epoch_val_loss / samples
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch:2d}: Train Loss={avg_train_loss:.2e}, Val Loss={avg_val_loss:.2e}")
    
    return train_losses, val_losses
        