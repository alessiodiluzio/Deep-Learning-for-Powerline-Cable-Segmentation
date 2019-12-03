import os

LABEL_PATH = ['../file/input/label/cable_blank', '../file/input/label/cable_depth',
              '../file/input/label/cable_horizontal', '../file/input/label/cable_rect']


def rename(label_path):
    for path in label_path:
        for r, d, images in os.walk(path):
            for img in images:
                os.rename(os.path.join(r, img), os.path.join(r, img.replace("_ground_truth", "")))


rename(LABEL_PATH)
