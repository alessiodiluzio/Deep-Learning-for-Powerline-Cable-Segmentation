import os

LABEL_PATH = '../file/input/label'


def rename(label_path):
    for r, d, images in os.walk(label_path):
        for img in images:
            os.rename(os.path.join(r, img), os.path.join(r, img.replace("rect", "")))


rename(LABEL_PATH)
