import os
import random
from shutil import copyfile


def get_perc_dataset(src_path, dest_path, perc):
    for r, d, images in os.walk(src_path):
        random.shuffle(images)
        tot = perc*len(images)
        i = 0
        for img in images:
            if i > tot:
                return
            i += 1
            copyfile(os.path.join(r, img), os.path.join(dest_path, img))
            label_path = get_label_path(os.path.join(r, img))
            dest_label_path = get_label_path(dest_path)
            copyfile(label_path, os.path.join(dest_label_path, img))


def get_label_path(image_path):
    return image_path.replace('normal', 'label')


def merge(src, dest):
    paths = ['cable_blank', 'cable_horizontal', 'cable_rect', 'cable_depth']
    for path in paths:
        complete_path = os.path.join(src, path)
        for r, d, images in os.walk(complete_path):
            for img in images:
                copyfile(os.path.join(r, img), os.path.join(dest, img))


# get_perc_dataset('../file/input/normal/cable_horizontal', '../file/input/small/normal/cable_horizontal', 0.1)
merge('../file/input/small/normal', '../file/input/small/all_small/normal')
