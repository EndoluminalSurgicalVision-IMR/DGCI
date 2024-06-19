# -*- coding: utf-8 -*-

import os

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



