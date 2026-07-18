# experiments/configs.py: batch experiment combinations for run.py

CONFIGS = [
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet18", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet34", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19", "head": "coord_spatial"},

    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "vit_b_16", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "swin_t", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "wide_resnet50_2.tv_in1k", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "deit_base_distilled_patch16_224.fb_in1k", "head": "coord_spatial"},
    # {"method": "reg", "batch_size": 4, "max_epochs": 5, "backbone": "cait_s24_224.fb_dist_in1k", "head": "coord_spatial"},

    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet18", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet34", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "efficientnet_b0", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "swin_t", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19", "head": "mask"},
    # {"method": "seg", "batch_size": 4, "max_epochs": 5, "backbone": "wide_resnet50_2.tv_in1k", "head": "mask"},

    # {"method": "seg", "model": "fcn_resnet50", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
    # {"method": "seg", "model": "deeplabv3_resnet50", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
    # {"method": "seg", "model": "deeplabv3_mobilenet_v3_large", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},
    # {"method": "seg", "model": "lraspp_mobilenet_v3_large", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "mask"},

    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "resnet18", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "resnet34", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "resnet50", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "efficientnet_b0", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "swin_t", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "vgg16", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "vgg19", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "wide_resnet50_2.tv_in1k", "head": "box"},
    # {"method": "det", "batch_size": 4, "max_epochs": 5, "backbone": "custom", "head": "point"},

    {"method": "det", "model": "fasterrcnn_resnet50_fpn", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
    {"method": "det", "model": "retinanet_resnet50_fpn", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
    {"method": "det", "model": "ssd300_vgg16", "batch_size": 4, "max_epochs": 5, "backbone": "", "head": "box"},
]
