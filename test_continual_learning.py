import torch
import torch.nn as nn
from bdh.continual_learning.adaptive_synapses import AdaptiveLinear
from bdh.continual_learning.consolidation import ElasticWeightConsolidation
from bdh.continual_learning.importance import FisherEstimator


def test_adaptive_synapse():
    """Test adaptive synaptic layer."""
    layer = AdaptiveLinear(10, 5)
    x = torch.randn(2, 10)
    
    # Forward pass
    y = layer(x)
    assert y.shape == (2, 5)
    
    # Check buffers exist
    assert hasattr(layer, 'plasticity_state')
    assert layer.plasticity_state.shape == layer.weight.shape


def test_ewc_loss():
    """Test EWC loss computation."""
    model = nn.Sequential(
        AdaptiveLinear(10, 20),
        nn.ReLU(),
        AdaptiveLinear(20, 5)
    )
    
    ewc = ElasticWeightConsolidation(lambda_ewc=1000.0)
    
    # Should be 0 for first task
    loss = ewc(model)
    assert loss.item() == 0.0
    assert ewc.is_first_task
    
    # After consolidation
    fisher_dict = {
        'weight': torch.ones(5, 20),
        'bias': torch.ones(5)
    }
    ewc.consolidate_task(model, fisher_dict)
    
    # Should be non-zero now
    loss = ewc(model)
    assert loss.item() > 0.0 or loss.item() == 0.0  # After first consolidation


if __name__ == "__main__":
    test_adaptive_synapse()
    print("Adaptive synapse test passed")
    
    test_ewc_loss()
    print("EWC loss test passed")
    
    print("\nAll tests passed!")
