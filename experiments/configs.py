# experiments/configs.py: batch experiment combinations for run.py

CONFIGS = [
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet18", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet34", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "efficientnet_b0", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16_bn", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19", "head": "coord_spatial"},
    {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19_bn", "head": "coord_spatial"},
]
