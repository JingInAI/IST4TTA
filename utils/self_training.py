import torch.nn as nn


class MomentumUpdate():
    def __init__(self, model: nn.Module = None, classifier: nn.Module = None):
        self.model_params = model.state_dict() if model is not None else None
        self.classifier_params = classifier.state_dict() if classifier is not None else None
        

    def __call__(self, model: nn.Module = None, classifier: nn.Module = None, m: int = 0.99):
        assert m >= 0 and m <= 1, "m should be in [0, 1]"

        if model is not None:
            new_model_params = model.state_dict()

            model_dict = {}
            for (name, params), (_, new_params) in zip(self.model_params.items(), new_model_params.items()):
                model_dict[name] = m * params.data.detach().clone() + (1. - m) * new_params.data.detach().clone()

            self.model_params = model_dict
            model.load_state_dict(model_dict)

        if classifier is not None:
            new_classifier_params = classifier.state_dict() if classifier is not None else None

            classifier_dict = {}
            for (name, params), (_, new_params) in zip(self.classifier_params.items(), new_classifier_params.items()):
                classifier_dict[name] = m * params.data.detach().clone() + (1. - m) * new_params.data.detach().clone()

            self.classifier_params = classifier_dict        
            classifier.load_state_dict(classifier_dict)

        return model, classifier



def freeze(model: nn.Module):
    """ freeze all parameters in model
    Note that this function will use model.eval() to freeze BatchNorm layers
    Args:
        model (nn.Module): model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def unfreeze(model: nn.Module):
    """ set "requires_grad=True" for all parameters in model
    Note that this function will use model.train() to unfreeze BatchNorm layers
    Args:
        model (nn.Module): model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def freeze_norm_layer(model: nn.Module):
    """ freeze all normalization layers in model
    Args:
        model (nn.Module): model to freeze
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            freeze(m)

def unfreeze_norm_layer(model: nn.Module):
    """ unfreeze all normalization layers in model
    Args:
        model (nn.Module): model to unfreeze
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            unfreeze(m)

def keep_norm_layer_unfreezed(model: nn.Module):
    """ keep all normalization layers in model unfreezed and freeze other layers
    Args:
        model (nn.Module): model to freeze
    """
    for name, param in model.named_parameters():
        if 'bn' in name or 'norm' in name:
            continue
        else:
            param.requires_grad = False


def init_classifier(m):
    """ initialize classifier layer
    Args:
        m (nn.Module): classifier layer
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def tent_freeze(model: nn.Module):
    """ freeze all parameters in model except Norm layers
    Args:
        model (nn.Module): model to freeze
    """
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def norm_adapt(model: nn.Module):
    """ freeze all parameters in model except forward statistics of Norm layers
    Args:
        model (nn.Module): model to freeze
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            m.train()
