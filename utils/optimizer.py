import torch
import math


def get_optimizer(model, opt_name, opt_kwargs):

    optimizer_init = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW
        }[opt_name.lower()]

    optimizer = optimizer_init(
        model.parameters(), 
        lr=opt_kwargs["lr"], 
        weight_decay=opt_kwargs["weight_decay"]
    )

    if opt_kwargs['scheduler']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=opt_kwargs['milestones'], 
                                                          gamma=opt_kwargs['gamma'])
        return optimizer, scheduler
    else:
        return optimizer
