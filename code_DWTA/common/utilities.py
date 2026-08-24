import logging
import os
import datetime
import pytz
import re
import math
import numpy as np
from .TORCH_OBJECTS import *
from .Dynamic_HYPER_PARAMETER import *


########################################
# Get_Logger
########################################
tz = pytz.timezone("America/Phoenix")

def calculate_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def timetz(*args):
    return datetime.datetime.now(tz).timetuple()


def Get_Logger(SAVE_FOLDER_NAME):
    # make_dir
    #######################################################
    prefix = datetime.datetime.now(pytz.timezone("America/Phoenix")).strftime("%Y%m%d_%H%M__")
    result_folder_no_postfix = "./result/{}".format(SAVE_FOLDER_NAME)

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)

    os.makedirs(result_folder_path)

    # Logger
    #######################################################
    logger = logging.getLogger(result_folder_path)  # this already includes streamHandler??

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('{}/log_2.txt'.format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)

    return logger, result_folder_path


def Extract_from_LogFile(result_folder_path, variable_name):
    logfile_path = '{}/log_2.txt'.format(result_folder_path)
    with open(logfile_path) as f:
        datafile = f.readlines()
    found = False  # This isn't really necessary
    for line in reversed(datafile):
        if variable_name in line:
            found = True
            m = re.search(variable_name + '[^\n]+', line)
            break
    exec_command = "Print(No such variable found !!)"
    if found:
        return m.group(0)
    else:
        return exec_command


########################################
# Average_Meter
########################################

class Average_Meter:

    def __init__(self):
        self.sum = None
        self.count = None
        self.reset()


    def reset(self):
        self.sum = torch.tensor(0.).to(DEVICE)
        self.count = 0

    def push(self, some_tensor, n_for_rank_0_tensor=None):
        assert not some_tensor.requires_grad  # You get Memory error, if you keep tensors with grad history

        rank = len(some_tensor.shape)

        if rank == 0:  # assuming "already averaged" Tensor was pushed
            self.sum += some_tensor * n_for_rank_0_tensor
            self.count += n_for_rank_0_tensor

        else:
            self.sum += some_tensor.sum()
            self.count += some_tensor.numel()

    def peek(self):
        average = (self.sum / self.count).tolist()
        return average

    def result(self):
        average = (self.sum / self.count).tolist()
        self.reset()
        return average


########################################
# View NN Parameters
########################################

def get_n_params1(model):
    pp = 0
    for p in list(model.parameters()):
        nn_count = 1
        for s in list(p.size()):
            nn_count = nn_count * s
        pp += nn_count
        print(nn_count)
        print(p.shape)
    print("Total: {:d}".format(pp))


def get_n_params2(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def get_n_params3(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_structure(model):
    print(model)

def convert_tensor_based_on_percentile(before_value, percentile):
    import random
    """
    Convert values in the tensor to 0, 1, or random based on the given percentile threshold.

    Parameters:
    - tensor (torch.Tensor): Input tensor of shape [N, 1].
    - percentile (float): Percentile threshold to use for conversion.

    Returns:
    - torch.Tensor: Output tensor with values converted to 0, 1, or random.
    """
    # Calculate the threshold value at the given percentile
    threshold = torch.quantile(before_value, percentile / 100.0)

    mask_greater = before_value > threshold
    mask_less = before_value < threshold
    mask_equal = before_value == threshold
    # Initialize the output tensor
    output_tensor = torch.empty_like(before_value)
    # Assign values based on the masks
    output_tensor[mask_greater] = 1
    output_tensor[mask_less] = 0
    output_tensor[mask_equal] = torch.tensor(random.choice([1, 0]), dtype=before_value.dtype)

    return output_tensor


########################################
# Coordination / soft-coupling metrics
#
# Operationalize "wasteful over-concentration of fire" (the failure mode
# the parallel decoder's soft-coupling hypothesis is about, per
# BReRLA_revision_plan.md item 11) from the shot log an eval rollout
# produces. Pure functions over already-extracted Python scalars -- no
# simulator changes needed, callers just log each shot's pre-shot target
# value fraction and target index during rollout.
########################################

def compute_overkill_rate(pre_shot_value_fractions, threshold=0.1):
    """
    Fraction of shots fired at a target whose remaining value, immediately
    before that shot, was already below `threshold` of its original value
    (i.e. the target was already close to fully destroyed -- diminishing
    returns per Eq. 1 -- so the shot was largely redundant).

    Args:
        pre_shot_value_fractions: list/iterable of floats in [0, 1], one per
            shot, each = (target's remaining value just before this shot) /
            (that target's original value).
        threshold: fraction below which a target is considered "already
            effectively dead" (default 0.1, matching the neutralization
            discussion in DISCUSSION_NOTES.md).

    Returns:
        float in [0, 1]: wasted_shots / total_shots. 0.0 if no shots fired.
    """
    fractions = list(pre_shot_value_fractions)
    if not fractions:
        return 0.0
    wasted = sum(1 for f in fractions if f < threshold)
    return wasted / len(fractions)


def compute_fire_dispersion(shot_counts_per_target):
    """
    Normalized entropy of shots-per-target within an episode, as a
    general shape-of-coordination indicator: 0.0 = every shot landed on a
    single target (fully concentrated), 1.0 = shots spread as evenly as
    possible across all targets that received at least one shot.

    Args:
        shot_counts_per_target: list/iterable of per-target shot counts
            (non-negative ints/floats), one entry per target.

    Returns:
        float in [0, 1]. 0.0 if no shots fired or only one target was
        ever engaged (entropy undefined / degenerate).
    """
    counts = [c for c in shot_counts_per_target if c > 0]
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0