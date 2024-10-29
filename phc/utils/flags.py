__all__ = ['flags', 'summation']

class Flags(object):
    def __init__(self, items):
        for key, val in items.items():
            setattr(self,key,val)

flags = Flags({
    'test': False, 
    'debug': False,
    "real_traj": False,
    "im_eval": False,
    "finetune_wRR": False,      # rendering reward
    "position_only": False,     # only 3D position
    "pos_2d_only": False,       # Change 3D position to 2D position
    "fix_disc": False,
    "lora": False,
    })
