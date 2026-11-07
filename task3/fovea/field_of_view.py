"""Label-independent complete-retina crops and original-pixel restoration."""
import numpy as np
from PIL import Image

from .data import restore_xy


BOX_COLUMNS = ['crop_left', 'crop_top', 'crop_right', 'crop_bottom']


def visible_field_box(image, threshold=20, padding=32):
    width, height = image.size
    small = np.asarray(image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR))
    yy, xx = np.where(small.max(axis=2) > threshold)
    if len(xx) == 0:
        raise ValueError('No visible retinal field')
    return [max(0, int(np.floor(xx.min()*width/512))-padding),
            max(0, int(np.floor(yy.min()*height/512))-padding),
            min(width, int(np.ceil((xx.max()+1)*width/512))+padding),
            min(height, int(np.ceil((yy.max()+1)*height/512))+padding)]


def prepare_field_frame(frame, threshold=20, padding=32):
    result = frame.copy(deep=True)
    boxes = []
    for path in frame.path:
        with Image.open(path) as image:
            boxes.append(visible_field_box(image, threshold, padding))
    boxes = np.asarray(boxes)
    result[['source_width', 'source_height']] = frame[['width', 'height']].to_numpy()
    result[BOX_COLUMNS] = boxes
    result[['width', 'height']] = boxes[:, 2:]-boxes[:, :2]
    if 'Fovea_X' in frame:
        result[['Fovea_X', 'Fovea_Y']] = frame[['Fovea_X', 'Fovea_Y']].to_numpy()-boxes[:, :2]
    return result


def restore_field_xy(xy, field_frame):
    return restore_xy(xy, field_frame[['width', 'height']])+field_frame[['crop_left', 'crop_top']].to_numpy()
