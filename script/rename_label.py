import os
from PIL import Image

LABEL_PATH = ['../file/input/label/cable_blank', '../file/input/label/cable_depth',
              '../file/input/label/cable_horizontal', '../file/input/label/cable_rect']


def rename(label_path):
    if not isinstance(label_path, list):
        label_path = [label_path]
    for path in label_path:
        for r, d, images in os.walk(path):
            for img in images:
                print(img)
                os.rename(os.path.join(r, img), os.path.join(r, img.replace("_ground_truth", "")))


def rename_test(test_path, label_path):
    for r, d, images in os.walk(test_path):
        for img in images:
            im = Image.open(os.path.join(r, img))
            name = img.split('.')[0]
            im.save(os.path.join(r, name + '.png'), 'PNG')

    for r, d, images in os.walk(label_path):
        for img in images:
            im = Image.open(os.path.join(r, img))
            name = img.split('.')[0]
            im.save(os.path.join(r, name + '.png'), 'PNG')

    for r, d, images in os.walk(test_path):
        for img in images:
            name = img.split('_')[3]
            os.rename(os.path.join(r, img), os.path.join(r, name))

    for r, d, images in os.walk(label_path):
        for img in images:
            name = img.split('_')[3]
            os.rename(os.path.join(r, img), os.path.join(r, name))


def delete(test_path, label_path):
    for r, d, images in os.walk(test_path):
        for img in images:
            if 'bmp' in img:
                os.remove(os.path.join(r, img))

    for r, d, images in os.walk(label_path):
        for img in images:
            if 'bmp' in img:
                os.remove(os.path.join(r, img))

rename("../file/input/large/label")
#rename(LABEL_PATH)
#rename_test(test_path='../file/input/test/normal', label_path='../file/input/test/label')
#delete(test_path='../file/input/test/normal', label_path='../file/input/test/label')