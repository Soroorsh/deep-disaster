import cv2
import numpy as np


def convert_to_grayscale(im_as_arr):
    """
    Convert an 3D numpy array (RGB) to a 2D one (grayscale)

    Args:
            im_as_arr: Input nd.Array

    Returns: 
            grayscale_im : nd.Array
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def get_threshold(scores, contamination_rate_estimation):
    """ Computes the fp = fn threshold.

    Args:
        scores: np.ndarray(Float); anomaly-scores to base the threshold on
        contamination_rate_estimation: Float; estimation of the contamination rate

    Returns:
        Float; fp = fn threshold
    """
    return np.percentile(scores, (1-contamination_rate_estimation)*100)


def show_localization_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the localization mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    Args:
        img: The base image in RGB or BGR format.
        mask: The localization mask.
        use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        colormap: The OpenCV colormap to be used.
    Returns: 
        The default image with the cam overlay.
    """
    height, width, _ = img.shape
    mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
    mask = cv2.resize(mask, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    loc = heatmap + img
    loc = loc / np.max(loc)
    return np.uint8(255 * loc)
