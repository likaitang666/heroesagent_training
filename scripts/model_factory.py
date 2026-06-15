"""模型工厂 — 构建并配置分类模型。

支持 torchvision 所有分类模型, 自动替换分类头以适配自定义类别数。

用法:
    from model_factory import build_model
    model = build_model("mobilenet_v3_large", num_classes=191)
"""

import torch.nn as nn


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """构建分类模型, 自动替换分类头。

    Args:
        model_name: torchvision模型名 (如 mobilenet_v3_large)
        num_classes: 分类数
        pretrained: 是否使用ImageNet预训练权重

    Returns:
        替换了分类头的模型

    Raises:
        ValueError: 模型名不存在或无法找到分类头
    """
    import torchvision.models as models

    model_func = getattr(models, model_name, None)
    if model_func is None:
        available = [m for m in dir(models)
                     if m[0].islower() and not m.startswith("_")]
        raise ValueError(f"未知模型: {model_name}\n可用: {', '.join(available)}")

    weights = "IMAGENET1K_V1" if pretrained else None
    model = model_func(weights=weights)

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"无法找到模型分类头: {model_name}")

    return model
