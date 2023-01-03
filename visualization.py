from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow


def plot_frames(frames, rows, cols):
    fig = plt.figure(figsize=(32, 8))
    for i, image in enumerate(frames):
        if i >= rows * cols:
            break
        ax = plt.subplot(rows, cols, i + 1)
        imshow(image)
