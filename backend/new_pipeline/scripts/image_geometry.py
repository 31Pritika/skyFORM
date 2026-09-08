"""Pixel transforms matching DUSt3R's 512px resize and centre crop."""
import numpy as np


def crop_transform(camera, width, height, image_size=512):
    ow, oh = camera['width'], camera['height']
    factor = image_size / max(ow, oh)
    rw, rh = round(ow * factor), round(oh * factor)
    left, top = rw // 2 - width / 2, rh // 2 - height / 2
    return rw / ow, rh / oh, left, top


def original_to_prediction(x, y, camera, width, height):
    sx, sy, left, top = crop_transform(camera, width, height)
    return (np.asarray(x) + .5) * sx - .5 - left, (np.asarray(y) + .5) * sy - .5 - top


def prediction_to_original(x, y, camera, width, height):
    sx, sy, left, top = crop_transform(camera, width, height)
    return (np.asarray(x) + left + .5) / sx - .5, (np.asarray(y) + top + .5) / sy - .5
