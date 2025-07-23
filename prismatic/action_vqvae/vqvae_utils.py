from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# def var(tensor):
#     return tensor.to(device)


def get_tensor(z, device):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return torch.FloatTensor(z.copy()).to(device).unsqueeze(0)
        # return torch.from_numpy(z.copy()).float().to(device).unsqueeze(0)
    else:
        return torch.FloatTensor(z.copy()).to(device)
        # return torch.from_numpy(z.copy()).float().to(device)


ROBOT_LABLES = [
    "google_robot",
    "kuka_iiwa",
    "widowx",
    "franka",
    "jaco_2",
    "sawyer",
    "ur5",
    "xarm",
    "dlr_edan",
    "fanuc_mate",
    "hello_stretch",
]

CONTROL_FREQUENCY = [2, 3, 3.75, 5, 10, 15, 20, 30]

OXE_ROBOT: Dict[bytes, Tuple[str, float]] = {
    b"fractal20220817_data": ("google_robot", 3),
    b"kuka": ("kuka_iiwa", 10),
    b"bridge_orig": ("widowx", 5),
    b"taco_play": ("franka", 15),
    b"jaco_play": ("jaco_2", 10),
    b"berkeley_cable_routing": ("franka", 10),
    b"roboturk": ("sawyer", 10),
    b"viola": ("franka", 20),
    b"berkeley_autolab_ur5": ("ur5", 5),
    b"toto": ("franka", 30),
    b"language_table": ("xarm", 10),
    b"stanford_hydra_dataset_converted_externally_to_rlds": ("franka", 10),
    b"austin_buds_dataset_converted_externally_to_rlds": ("franka", 20),
    b"nyu_franka_play_dataset_converted_externally_to_rlds": ("franka", 3),
    b"furniture_bench_dataset_converted_externally_to_rlds": ("franka", 10),
    b"ucsd_kitchen_dataset_converted_externally_to_rlds": ("xarm", 2),
    b"austin_sailor_dataset_converted_externally_to_rlds": ("franka", 20),
    b"austin_sirius_dataset_converted_externally_to_rlds": ("franka", 20),
    b"dlr_edan_shared_control_converted_externally_to_rlds": ("dlr_edan", 5),
    b"iamlab_cmu_pickup_insert_converted_externally_to_rlds": ("franka", 20),
    b"utaustin_mutex": ("franka", 20),
    b"berkeley_fanuc_manipulation": ("fanuc_mate", 10),
    b"cmu_stretch": ("hello_stretch", 10),
    b"bc_z": ("google_robot", 10),  # Note: (use v0. --> later versions broken
    b"fmb_dataset": ("franka", 10),
    b"dobbe": ("hello_stretch", 3.75),
    b"libero_spatial_no_noops": ("franka", 20),
    b"libero_object_no_noops": ("franka", 20),
    b"libero_goal_no_noops": ("franka", 20),
    b"libero_10_no_noops": ("franka", 20),
    b"libero_90_no_noops": ("franka", 20),
    b"maniskill_dataset_converted_externally_to_rlds": ("franka", 20),
    b"droid": ("franka", 20),
}
