#!/usr/bin/env python

import numpy as np






def get_wb(model, in_dataset, m_name):

    model.eval()
    cache_name = f"cache/{in_dataset}_{m_name}_in_alllayers.npy"
    w = model.fc.weight.cpu().detach().squeeze().numpy()
    b = model.fc.bias.cpu().detach().squeeze().numpy()
    #np.save(cache_name, [w,b])
    return w,b




