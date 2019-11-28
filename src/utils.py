import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime


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
    return sum(1 for _ in tf.data.TFRecordDataset(tf_record_path))


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
            if 'BLACK' in line:
                perc_black = line.split('=')[1]
            if 'WHITE' in line:
                perc_white = line.split('=')[1]
    return float(perc_black), float(perc_white)


def plot_metrics(model_history, epochs, save_path, validation=False):
    epochs = range(epochs)
    metrics = []
    for key in model_history:
        if validation:
            if 'val' in key:
                metrics.append(key)
        else:
            if 'val' not in key:
                metrics.append(key)
    for i in range(0, len(metrics)):
        metric = metrics[i]
        m = model_history[metric]
        label = 'Training'
        if validation:
            label = 'Validation'
        plt.figure()
        color = 'red'
        if validation:
            color = 'blue'
        plt.plot(epochs, m, color=color, linestyle='-', label=label + ' ' + metric)
        plt.title(label + " " + metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric+' Value')
        plt.ylim([0, 2])
        plt.legend()
        plt.savefig(os.path.join(save_path, label + '_' + metric + '.jpg'))
        plt.show()


def create_folder_and_save_path(dir_path, model_name):
    folder_name = model_name + "_" + str(datetime.datetime.now()).replace(':', '_')
    folder_path = dir_path + folder_name
    os.mkdir(folder_path)
    os.mkdir(folder_path + '/Training')
    os.mkdir(folder_path + '/Validation')
    return folder_path


def plot(dir_path, model_name, model_history, epochs,):
    save_path = create_folder_and_save_path(dir_path, model_name)
    plot_metrics(model_history, epochs, save_path + '/Training', validation=False)
    plot_metrics(model_history, epochs, save_path + '/Validation', validation=True)
