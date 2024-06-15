from .resnet import *
from .wide_resnet import *
from .linear import *


def load_model(name, pretrained=True, device='cuda', **kwargs):
    if name == 'resnet18':
        backbone = resnet18(pretrained=pretrained, **kwargs)
        fc = linear(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    elif name == 'resnet34':
        backbone = resnet34(pretrained=pretrained, **kwargs)
        fc = linear(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    elif name == 'resnet50':
        backbone = resnet50(pretrained=pretrained, **kwargs)
        fc = linear(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    elif name == 'resnet101':
        backbone = resnet101(pretrained=pretrained, **kwargs)
        fc = linear(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    elif name == 'resnet152':
        backbone = resnet152(pretrained=pretrained, **kwargs)
        fc = linear(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    elif name == 'WRN-40-2':
        return wrn_40_2(pretrained=pretrained, **kwargs).to(device)

    elif name == 'resnet18_fc':
        return resnet18(pretrained=pretrained, with_fc=True, **kwargs).to(device)

    elif name == 'resnet50_shot':
        backbone = resnet50(pretrained=pretrained, shot_model=True, **kwargs)
        fc = linear_wn(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    elif name == 'resnet101_shot':
        backbone = resnet101(pretrained=pretrained, shot_model=True, **kwargs)
        fc = linear_wn(in_dim=backbone.num_features, out_dim=backbone.num_classes, **kwargs)
        return backbone.to(device), fc.to(device)

    else:
        raise NotImplementedError(f"model: {name} is not implemented.")
