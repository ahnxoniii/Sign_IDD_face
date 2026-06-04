# coding: utf-8
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
import yaml
import torch
import numpy as np

from torch import nn, Tensor
from dtw import dtw
from logging import Logger
from typing import Optional

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :param model_continue: whether to continue from a checkpoint
    :return: path to model directory
    """
    # If model already exists
    if os.path.isdir(model_dir):

        # If model continuing from checkpoint
        if model_continue:
            # Return the model_dir
            return model_dir

        # If set to not overwrite, this will error
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")

        # If overwrite, recursively delete previous directory to start with empty dir again
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)

    # If model directly doesn't exist, make it and return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(
        "{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production")
    return logger

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')

    return torch.from_numpy(mask) == 0 # Turns it into True and False's

# Subsequent mask of two sizes
def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # Turns it into True and False's

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")

def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fixe: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint

def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    # hypotheses = hypotheses[:, 1:]    # Non-autoregressive annotation

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores

face_skeletons = (
    # ----- new face root at nose 29 -----
    (1, 79),      # body joint 1 -> face global 79  (fa local 29)

  # ----- nose bridge / nose -----
    (79, 78),     # fa 29 -> 28
    (78, 77),     # fa 28 -> 27

    (79, 80),     # fa 29 -> 30
    (80, 83),     # fa 30 -> 33
    (83, 82),     # fa 33 -> 32
    (82, 81),     # fa 32 -> 31
    (83, 84),     # fa 33 -> 34
    (84, 85),     # fa 34 -> 35

    # ----- left eyebrow (fa 17~21 / global 67~71) -----
    (77, 71),     # fa 27 -> fa 21
    (71, 70),     # fa 21 -> fa 20
    (70, 69),     # fa 20 -> fa 19
    (69, 68),     # fa 19 -> fa 18
    (68, 67),     # fa 18 -> fa 17

    # ----- right eyebrow (fa 22~26 / global 72~76) -----
    (77, 72),     # fa 27 -> fa 22
    (72, 73),     # fa 22 -> fa 23
    (73, 74),     # fa 23 -> fa 24
    (74, 75),     # fa 24 -> fa 25
    (75, 76),     # fa 25 -> fa 26

    # ----- left eye (fa 36~41 / global 86~91) -----
    (71, 89),     # face global 71 (fa 21) -> face global 89 (fa 39)  # inner brow -> inner eye
    (89, 88),     # fa 39 -> 38
    (88, 87),     # fa 38 -> 37
    (87, 86),     # fa 37 -> 36
    (89, 90),     # fa 39 -> 40
    (90, 91),     # fa 40 -> 41

    # ----- right eye (fa 42~47 / global 92~97) -----
    (72, 92),     # face global 72 (fa 22) -> face global 92 (fa 42)  # inner brow -> inner eye
    (92, 93),     # fa 42 -> 43
    (93, 94),     # fa 43 -> 44
    (94, 95),     # fa 44 -> 45
    (92, 97),     # fa 42 -> 47
    (97, 96),     # fa 47 -> 46

    # ----- outer mouth (fa 48~59 / global 98~109) -----
    (83, 101),    # fa 33 -> fa 51

    # left branch
    (101, 100),   # fa 51 -> 50
    (100, 99),    # fa 50 -> 49
    (99, 98),     # fa 49 -> 48

    # right branch
    (101, 102),   # fa 51 -> 52
    (102, 103),   # fa 52 -> 53
    (103, 104),   # fa 53 -> 54

    # lower branch
    (101, 107),   # fa 51 -> 57
    (107, 106),   # fa 57 -> 56
    (106, 105),   # fa 56 -> 55
    (107, 108),   # fa 57 -> 58
    (108, 109),   # fa 58 -> 59

    # ----- inner mouth (fa 60~67 / global 110~117) -----
    (101, 112),   # face global 101 (fa 51) -> face global 112 (fa 62)
    (112, 111),   # face global 112 (fa 62) -> face global 111 (fa 61)
    (111, 110),   # face global 111 (fa 61) -> face global 110 (fa 60)
    (112, 113),   # face global 112 (fa 62) -> face global 113 (fa 63)
    (113, 114),   # face global 113 (fa 63) -> face global 114 (fa 64)
    (114, 115),   # face global 114 (fa 64) -> face global 115 (fa 65)
    (115, 116),   # face global 115 (fa 65) -> face global 116 (fa 66)
    (116, 117),   # face global 116 (fa 66) -> face global 117 (fa 67)

    # ----- jaw (fa 0~16 / global 50~66) -----
    (107, 58),    # face global 107 (fa 57) -> face global 58 (fa 8)

    (58, 57),     # fa 8 -> 7
    (57, 56),     # fa 7 -> 6
    (56, 55),     # fa 6 -> 5
    (55, 54),     # fa 5 -> 4
    (54, 53),     # fa 4 -> 3
    (53, 52),     # fa 3 -> 2
    (52, 51),     # fa 2 -> 1
    (51, 50),     # fa 1 -> 0

    (58, 59),     # fa 8 -> 9
    (59, 60),     # fa 9 -> 10
    (60, 61),     # fa 10 -> 11
    (61, 62),     # fa 11 -> 12
    (62, 63),     # fa 12 -> 13
    (63, 64),     # fa 13 -> 14
    (64, 65),     # fa 14 -> 15
    (65, 66),     # fa 15 -> 16
)

face_skeletons_cvpr = (

# ----- face outline: top center 174 -> both sides -> chin center 68 -----

# ----- new face root -----
(1, 174),     # body head/root -> face top center

# left side: 174 -> ... -> 68
(174, 154),
(154, 110),
(110, 109),
(109, 136),
(136, 99),
(99, 98),
(98, 139),
(139, 76),
(76, 75),
(75, 137),
(137, 135),
(135, 134),
(134, 145),
(145, 177),
(177, 118),
(118, 117),
(117, 132),
(132, 68),

# right side: 174 -> ... -> 68
(174, 126),
(126, 127),
(127, 96),
(96, 97),
(97, 150),
(150, 151),
(151, 169),
(169, 114),
(114, 115),
(115, 116),
(116, 171),
(171, 133),
(133, 88),
(88, 89),
(89, 93),
(93, 94),
(94, 67),
(67, 68),

# ----- right eyebrow upper -----
(174, 153),
(153, 152),
(152, 78),
(78, 77),
(77, 147),

# ----- right eyebrow lower -----
(153, 122),
(122, 121),
(121, 144),
(144, 143),
(143, 161),

# ----- left eyebrow upper -----
(174, 70),
(70, 69),
(69, 146),
(146, 57),
(57, 56),

# ----- left eyebrow lower -----
(70, 176),
(176, 155),
(155, 112),
(112, 111),
(111, 158),

# ----- right eye: inner/glabella side -> outward -----
(122, 60),     # right eyebrow lower inner anchor -> eye inner anchor

(60, 170),
(170, 131),
(131, 102),
(102, 101),
(101, 138),
(138, 72),
(72, 71),
(71, 100),

(60, 53),
(53, 52),
(52, 159),
(159, 85),
(85, 66),
(66, 65),
(65, 156),

# ----- left eye: inner/glabella side -> outward -----
(176, 120),    # left eyebrow lower inner anchor -> eye inner anchor

# upper/outward branch
(120, 90),
(90, 74),
(74, 73),
(73, 149),
(149, 108),
(108, 107),
(107, 160),
(160, 103),    # temple side

# lower/outward branch
(120, 119),
(119, 157),
(157, 148),
(148, 64),
(64, 63),
(63, 83),
(83, 104),

# ----- outer mouth: tree version, spread from 128 and reach upper lip center -----

(68, 128),    # chin center -> lower lip center

# left branch: 128 -> ... -> 82
(128, 142),
(142, 141),
(141, 173),
(173, 164),
(164, 105),
(105, 106),
(106, 124),
(124, 125),
(125, 81),
(81, 82),

# right branch: 128 -> ... -> 80
(128, 129),
(129, 168),
(168, 113),
(113, 79),
(79, 80),
(80, 87),
(87, 86),
(86, 163),
(163, 162),

# ----- inner mouth: outer lower center -> inner lower center, spread both sides -----

(128, 59),    # outer lower center -> inner lower/inside center

# one side: 59 -> ... -> 62
(59, 58),
(58, 140),
(140, 175),
(175, 172),
(172, 166),
(166, 167),
(167, 61),
(61, 62),

# other side: 59 -> ... -> 95
(59, 123),
(123, 165),
(165, 91),
(91, 92),
(92, 95),
(95, 130),
(130, 55),
(55, 54),
(54, 84),
(84, 51),
(51, 50),

)


face_skeletons_cvpr = (
    # =====================================================
    # Face outline independent
    # root: 173
    # 173에서 양갈래로 퍼짐
    # =====================================================

    (173, 173),

    # branch 1: 173 -> ... -> 67
    (173, 153),
    (153, 109),
    (109, 108),
    (108, 135),
    (135, 98),
    (98, 97),
    (97, 138),
    (138, 75),
    (75, 74),
    (74, 136),
    (136, 134),
    (134, 133),
    (133, 144),
    (144, 176),
    (176, 117),
    (117, 116),
    (116, 131),
    (131, 67),

    # branch 2: 173 -> ... opposite outline
    (173, 125),
    (125, 126),
    (126, 95),
    (95, 96),
    (96, 149),
    (149, 150),
    (150, 168),
    (168, 113),
    (113, 114),
    (114, 115),
    (115, 170),
    (170, 132),
    (132, 87),
    (87, 88),
    (88, 92),
    (92, 93),
    (93, 66),

    # =====================================================
    # Right eyebrow independent
    # root: 152
    # =====================================================
    (152, 152),

    # upper
    (152, 151),
    (151, 77),
    (77, 76),
    (76, 146),

    # lower
    (152, 121),
    (121, 120),
    (120, 143),
    (143, 142),
    (142, 160),

    # =====================================================
    # Left eyebrow independent
    # root: 69
    # =====================================================
    (69, 69),

    # upper
    (69, 68),
    (68, 145),
    (145, 56),
    (56, 55),

    # lower
    (69, 175),
    (175, 154),
    (154, 111),
    (111, 110),
    (110, 157),

    # =====================================================
    # Right eye independent
    # root: 59
    # =====================================================
    (59, 59),

    # upper/outward branch
    (59, 169),
    (169, 130),
    (130, 101),
    (101, 100),
    (100, 137),
    (137, 71),
    (71, 70),
    (70, 99),

    # lower/outward branch
    (59, 52),
    (52, 51),
    (51, 158),
    (158, 84),
    (84, 65),
    (65, 64),
    (64, 155),

    # =====================================================
    # Left eye independent
    # root: 119
    # =====================================================
    (119, 119),

    # upper/outward branch
    (119, 89),
    (89, 73),
    (73, 72),
    (72, 148),
    (148, 107),
    (107, 106),
    (106, 159),
    (159, 102),

    # lower/outward branch
    (119, 118),
    (118, 156),
    (156, 147),
    (147, 63),
    (63, 62),
    (62, 82),
    (82, 103),

    # =====================================================
    # Outer mouth independent
    # root: 81 = upper lip center
    # =====================================================
    (81, 81),

    # one side: 81 -> ... -> 127
    (81, 80),
    (80, 124),
    (124, 123),
    (123, 105),
    (105, 104),
    (104, 163),
    (163, 172),
    (172, 140),
    (140, 141),
    (141, 127),

    # other side: 81 -> ... -> 128
    (81, 161),
    (161, 162),
    (162, 85),
    (85, 86),
    (86, 79),
    (79, 78),
    (78, 112),
    (112, 167),
    (167, 128),

    # =====================================================
    # Inner mouth independent
    # root: 50 = inner upper lip center
    # =====================================================
    (50, 50),

    # one side: 50 -> ... -> 58
    (50, 83),
    (83, 53),
    (53, 54),
    (54, 129),
    (129, 94),
    (94, 91),
    (91, 90),
    (90, 164),
    (164, 122),
    (122, 58),

    # other side: 50 -> ... -> 57
    (50, 49),
    (49, 61),
    (61, 60),
    (60, 166),
    (166, 165),
    (165, 171),
    (171, 174),
    (174, 139),
    (139, 57),
)

def getSkeletalModelStructure():
    return (
        # Neck root
        (0, 0),   

        # left arm
        (0, 1),
        (1, 2),
        (2, 3),

        # right arm
        (0, 4),
        (4, 5),
        (5, 6),

        # right 손목 -> 연결
        (6, 7),   

        # right 손 엄지
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),

        # right 손 검지
        (7, 12),
        (12, 13),
        (13, 14),
        (14, 15),

        # right 손 중지
        (7, 16),
        (16, 17),
        (17, 18),
        (18, 19),

        # right 손 약지
        (7, 20),
        (20, 21),
        (21, 22),
        (22, 23),

        # right 손 소지
        (7, 24),
        (24, 25),
        (25, 26),
        (26, 27),


        # left hand: connected from left wrist 3
        (3, 28),

        #left 손 엄지
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),

        #left 손 검지
        (28, 33),
        (33, 34),
        (34, 35),
        (35, 36),

        #left 손 중지
        (28, 37),
        (37, 38),
        (38, 39),
        (39, 40),

        #left 손 약지
        (28, 41),
        (41, 42),
        (42, 43),
        (43, 44),

        #left 손 소지
        (28, 45),
        (45, 46),
        (46, 47),
        (47, 48),
        
        *face_skeletons_cvpr
    )