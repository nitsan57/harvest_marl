import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def save_img(rgb_arr, path, name):
    plt.imshow(rgb_arr, interpolation='nearest')
    plt.savefig(path + name)


def make_video_from_image_dir(vid_path, img_folder, video_name='trajectory', fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(
        rgb_imgs, vid_path, video_name=video_name, fps=fps)


def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                             fps=15, format="mp4v", resize=(640, 480)):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in tqdm(enumerate(rgb_arrs), desc="Video rendering"):
        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        video.write(image)

    video.release()
    cv2.destroyAllWindows()


def ascii_to_numpy(ascii_list):
    """converts a list of strings into a numpy array


    Parameters
    ----------
    ascii_list: list of strings
        List describing what the map should look like
    Returns
    -------
    arr: np.ndarray
        numpy array describing the map with ' ' indicating an empty space
    """

    arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            arr[row, col] = ascii_list[row][col]
    return arr


def map_to_colors(map, color_map):
    """Converts a map to an array of RGB values.
    Parameters
    ----------
    map: np.ndarray
        map to convert to colors
    color_map: dict
        mapping between array elements and desired colors
    Returns
    -------
    arr: np.ndarray
        3-dim numpy array consisting of color map
    """
    rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=int)
    for row_elem in range(map.shape[0]):
        for col_elem in range(map.shape[1]):
            rgb_arr[row_elem, col_elem,
                    :] = color_map[map[row_elem, col_elem]]

    return rgb_arr


def return_view(grid, pos, row_size, col_size):
    """Given a map grid, position and view window, returns correct map part

    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros

    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: list
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension

    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge,
                                               top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size: x + col_size + 1,
                   y - row_size: y + row_size + 1]
    return view


def pad_if_needed(left_edge, right_edge, top_edge, bot_edge, matrix):
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, top_pad


def pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
    pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                     'constant', constant_values=(const_val, const_val))
    return pad_mat
