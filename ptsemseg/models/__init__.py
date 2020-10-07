import copy

from ptsemseg.models.hardnet import hardnet, HardnetLowResolution


models = {
    "hardnet" : hardnet,
    "hardnet_lr": HardnetLowResolution
}


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return models[name]
    except:
        raise ("Model {} not available".format(name))
