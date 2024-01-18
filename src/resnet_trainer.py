from .base_trainer import BaseTrainer
from torchvision import models
import torch.nn as nn


class ResNet18Trainer(BaseTrainer):
    def __init__(self, train_loader, test_loader, validation_loader, num_classes):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers_to_extract = ['layer1', 'layer2', 'layer3', 'layer4']

        # Replace the last fully connected layer with a new linear layer
        in_features = model.fc.in_features
        # Adjust the output size to num_classes
        model.fc = nn.Linear(in_features, num_classes)

        super().__init__(model, train_loader, test_loader,
                         validation_loader, layers_to_extract)


class ResNet34Trainer(BaseTrainer):
    def __init__(self, train_loader, test_loader, validation_loader, num_classes):
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        layers_to_extract = ['layer1', 'layer2', 'layer3', 'layer4']

        # Replace the last fully connected layer with a new linear layer
        in_features = model.fc.in_features
        # Adjust the output size to num_classes
        model.fc = nn.Linear(in_features, num_classes)

        super().__init__(model, train_loader, test_loader,
                         validation_loader, layers_to_extract)


class ResNet50Trainer(BaseTrainer):
    def __init__(self, train_loader, test_loader, validation_loader, num_classes):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        layers_to_extract = ['layer1', 'layer2', 'layer3', 'layer4']

        # Replace the last fully connected layer with a new linear layer
        in_features = model.fc.in_features
        # Adjust the output size to num_classes
        model.fc = nn.Linear(in_features, num_classes)

        super().__init__(model, train_loader, test_loader,
                         validation_loader, layers_to_extract)


class ResNet101Trainer(BaseTrainer):
    def __init__(self, train_loader, test_loader, validation_loader, num_classes):
        model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2)
        layers_to_extract = ['layer1', 'layer2', 'layer3', 'layer4']

        # Replace the last fully connected layer with a new linear layer
        in_features = model.fc.in_features
        # Adjust the output size to num_classes
        model.fc = nn.Linear(in_features, num_classes)

        super().__init__(model, train_loader, test_loader,
                         validation_loader, layers_to_extract)


class ResNet152Trainer(BaseTrainer):
    def __init__(self, train_loader, test_loader, validation_loader, num_classes):
        model = models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V2)
        layers_to_extract = ['layer1', 'layer2', 'layer3', 'layer4']

        # Replace the last fully connected layer with a new linear layer
        in_features = model.fc.in_features
        # Adjust the output size to num_classes
        model.fc = nn.Linear(in_features, num_classes)

        super().__init__(model, train_loader, test_loader,
                         validation_loader, layers_to_extract)
