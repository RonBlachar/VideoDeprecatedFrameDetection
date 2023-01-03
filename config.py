import pathlib

from models import CAE, ConvLSTMAE

DATA_PATHS = {
    "train": pathlib.PosixPath('./datasets/train/videos/'),
    "test": pathlib.PosixPath(
        './datasets/test/videos/'),
    "val": pathlib.PosixPath(
        './datasets/val/videos/')
}

MODELS = {
    'CAE': CAE,
    'ConvLSTMAE': ConvLSTMAE
}

MODELS_PATHS = {
    'CAE': 'CAE_full_model',
    'ConvLSTMAE': 'ConvLSTMAE_full_model',
}

NOISE_TYPES = [
    'white_noise',
    'gaussian',
    'left_half_white_noise'
]

N_FRAMES = 10

NOISE_VIDEO_PROBABILITY = 0.5
NOISE_VIDEO_PROBABILITY_TEST = 0.5
NOISE_VIDEO_PROBABILITY_VAL = 0.4
