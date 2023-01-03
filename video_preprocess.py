import os
import random
import cv2
import numpy as np
import tensorflow as tf
import visualization
from skimage.util import random_noise

import config as conf

np.random.seed(0)


def format_frames(frame, output_size):
    """
      Pad and resize an image from a video.
      Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.
      Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


# TODO - convert to numpy arrays instead of lists
def frames_from_video_file(video_path, n_frames, output_size=(224, 224), add_random_noise=False):
    """
    Creates frames from each video file present for each category.
    if video has less than n_frames returns [],[]
    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.
      add_random_noise: Boolean to determine whether to noise a frame.
    Return: Tuple
      (
          np. array of frames; shape=(n_frames, height, width, channels),
          target,
          target labels - noise type per target
      )
      """
    # Read each video frame by frame
    result = []
    target = []
    target_labels = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    if video_length < n_frames:
        return [], []
    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    noisy_frame_idx = np.random.randint(0, n_frames - 1)
    for i in range(n_frames):
        ret, frame = src.read()
        curr_target = 0
        curr_target_label = 'OriginalData'
        if add_random_noise and i == noisy_frame_idx:
            noise_type = np.random.choice(conf.NOISE_TYPES, 1)[0]
            frame = noise_generator(frame, noise_type)
            curr_target = 1
            curr_target_label = noise_type
        frame = format_frames(frame, output_size)
        result.append(frame)
        target.append(curr_target)
        target_labels.append(curr_target_label)
    src.release()
    result = np.array(result)
    return result, target, target_labels


def noise_generator(frame, noise_type):
    noisy_image = []
    if noise_type == 'white_noise':
        noisy_image = np.random.random(frame.shape)
    elif noise_type == 'gaussian':
        noisy_image = random_noise(frame, noise_type, var=0.3, clip=True)
    elif noise_type == 'left_half_white_noise':
        noisy_image = frame
        white_noise_shape = (noisy_image.shape[0], noisy_image.shape[1] // 2, noisy_image.shape[2])
        left_half_white_noise = np.random.random(white_noise_shape)
        noisy_image[0:white_noise_shape[0], 0:white_noise_shape[1], 0:white_noise_shape[2]] = left_half_white_noise
    return noisy_image


class FrameGenerator:
    def __init__(self, path, n_frames, training=False, noisy_video_probability=0.2):
        """ Returns a set of frames with their associated label.

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.noisy_video_probability = noisy_video_probability

    def get_files(self, num_videos=-1):
        video_paths = list(self.path.glob('*.mov'))
        return video_paths if num_videos == -1 else video_paths[:num_videos]

    def generate_frames_array(self, num_videos=-1, stack_videos=False):
        video_paths = self.get_files(num_videos)
        frames_list = []
        target = []
        target_labels = []
        for i, path in enumerate(video_paths):
            is_noisy_video = (not self.training) * (random.random() < self.noisy_video_probability)
            video_frames, video_target, video_target_label = frames_from_video_file(path,
                                                                                    self.n_frames,
                                                                                    add_random_noise=is_noisy_video)
            if len(video_frames) == self.n_frames:
                frames_list.append(video_frames)
                target += video_target
                target_labels += video_target_label
        frames = np.stack(frames_list) if stack_videos else np.concatenate(frames_list)
        return frames, np.array(target), np.array(target_labels)


# def convert_avi_to_mp4(avi_file_path, output_name):
#     os.popen(
#         "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
#             input=avi_file_path, output=output_name))
#     return True
#
