# sota_models.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm

class SOTAModelWrapper:
    """
    Wrapper for state-of-the-art lightweight segmentation models
    """
    
    @staticmethod
    def create_mobilenet_deeplabv3(num_classes=1):
        """MobileNetV3 + DeepLabV3"""
        model = smp.DeepLabV3(
            encoder_name="mobilenet_v3_large",
            encoder_weights="imagenet",
            classes=num_classes,
            activation=None,
        )
        return model
    
    @staticmethod
    def create_efficientnet_deeplabv3plus(num_classes=1):
        """EfficientNet-B0 + DeepLabV3+"""
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            classes=num_classes,
            activation=None,
        )
        return model
    
    @staticmethod
    def create_efficientvit(num_classes=1):
        """EfficientViT (if available)"""
        # Note: Official EfficientViT may need custom implementation
        # Using simplified version for comparison
        from models.barm_efficientvit import BoundaryAwareEfficientViT
        model = BoundaryAwareEfficientViT(
            variant="b0",
            num_classes=num_classes,
            use_barm=False  # Baseline EfficientViT
        )
        return model
    
    @staticmethod
    def create_bisenetv2(num_classes=19):
        """BiSeNet V2"""
        model = smp.BiSeNetV2(
            encoder_name=None,  # No encoder for BiSeNetV2
            classes=num_classes,
            activation=None,
        )
        return model

# Helper function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# Example usage
if __name__ == "__main__":
    models = {
        "MobileNetV3+DeepLabV3": SOTAModelWrapper.create_mobilenet_deeplabv3(),
        "EfficientNet-B0+DeepLabV3+": SOTAModelWrapper.create_efficientnet_deeplabv3plus(),
        "EfficientViT-B0": SOTAModelWrapper.create_efficientvit(),
    }
    
    print("SOTA Models Comparison:")
    print("="*60)
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name:<35} | {params:>6.2f}M parameters")
    print("="*60)