import tensorflow as tf
import matplotlib.pyplot as plt
import os


def create_label_mask(label_mask):
    label_mask = tf.argmax(label_mask, axis=-1)
    label_mask = label_mask[..., tf.newaxis]
    return label_mask


def display_image(display_list):
    plt.figure(figsize=(30, 30))
    title = ['Image', 'Mask', "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def get_images(path):
    array = []
    for r, d, images in os.walk(path):
        for img in images:
            array.append(os.path.join(r, img))
    return array


def tf_record_count(tf_record_path):
    return sum(1 for _ in tf.compat.v1.io.tf_record_iterator(tf_record_path))


def get_mask_paths(training_paths):
    mask_paths = []
    for path in training_paths:
        mask_path = path.replace('normal', 'label')
        mask_paths.append(mask_path)
    return mask_paths


def read_pixel_frequency(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'BLACK' in line :
                perc_black = line.split('=')[1]
            if 'WHITE' in line :
                perc_white = line.split('=')[1]
    return perc_black, perc_white
