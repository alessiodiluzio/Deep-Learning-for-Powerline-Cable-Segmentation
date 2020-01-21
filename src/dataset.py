from PIL import Image, ImageEnhance
import tensorflow as tf
import random
import numpy as np


class DataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems."""

    def __init__(self, tf_records_path, image_size, palette=(0, 1)):
        """
        Initializes the data loader object
        Args:
            tf_records_path: Path to the TFRecord file for images and masks dataset
            palette: A list of RGB pixel values in the mask. If specified, the mask
                     will be one hot encoded along the channel dimension.
        """
        self.tf_records_path = tf_records_path
        self.palette = palette
        self.image_size = image_size

    @staticmethod
    def _tf_records_parser(example):
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'mask_raw': tf.io.FixedLenFeature([], tf.string),

        }
        image_features = tf.io.parse_single_example(example, image_feature_description)
        image_raw = image_features['image_raw']
        images = tf.io.decode_png(image_raw, channels=3)
        mask_raw = image_features['mask_raw']
        masks = tf.io.decode_png(mask_raw, 1)
        return images, masks

    @staticmethod
    def _resize_data(images, masks):
        """
        Resizes images to specified size.
        """
        images = tf.image.resize(images, [128, 128])
        masks = tf.image.resize(masks, [128, 128])
        return images, masks

    def _luminosity(self, images, masks):
        seed = random.random()
        images = tf.image.random_brightness(images, max_delta=0.5)
        images = tf.image.random_contrast(images, lower=1, upper=3.5)
        return images, masks

    @staticmethod
    def _normalize(images, masks):
        images = tf.cast(images, tf.float32) / 255.0
        masks = tf.cast(masks, tf.float32) / 255.0
        masks = tf.cast(masks, tf.int32)
        return images, masks

    def data_all(self):
        raw_image_dataset = tf.data.TFRecordDataset(self.tf_records_path)
        for example in raw_image_dataset:
            image_features = tf.io.parse_single_example(example, self.image_feature_description)
        # Parse images and labels
        data = raw_image_dataset.map(self._tf_records_parser)
        return data

    def validation_batch(self, val_batch_size):
        raw_image_dataset = tf.data.TFRecordDataset(self.tf_records_path)
        # Parse images and labels
        data = raw_image_dataset.map(self._tf_records_parser)
        data = data.map(self._resize_data)
        data = data.map(self._normalize)
        data = data.batch(val_batch_size, drop_remainder=True)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data


    def train_batch(self, batch_size, shuffle=False, augmentation=False):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            batch_size: Number of images/masks in each batch returned.
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
        Returns:
            data: A tf dataset object.
        """

        raw_image_dataset = tf.data.TFRecordDataset(self.tf_records_path)
        # Parse images and labels
        data = raw_image_dataset.map(self._tf_records_parser)
        if augmentation:
            data = data.map(self._luminosity)
        data = data.map(self._resize_data)
        data = data.map(self._normalize)
        if shuffle:
            data = data.shuffle(shuffle)
        data = data.batch(batch_size, drop_remainder=True)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data


class TestDataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems."""

    def __init__(self, tf_records_path, image_size, palette=(0, 1)):
        """
        Initializes the data loader object
        Args:
            tf_records_path: Path to the TFRecord file for images and masks dataset
            palette: A list of RGB pixel values in the mask. If specified, the mask
                     will be one hot encoded along the channel dimension.
        """
        self.tf_records_path = tf_records_path
        self.palette = palette
        self.image_size = image_size
        self.channels = []
        self.image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),

        }

    def _tf_records_parser(self, example):

        image_features = tf.io.parse_single_example(example, self.image_feature_description)
        image_raw = image_features['image_raw']
        images = tf.io.decode_png(image_raw, channels=self.channels[0])
        images = self._resize_data(images)
        images = self._normalize(images)
        return images

    def _resize_data(self, images):
        """
        Resizes images to specified size.
        """
        images = tf.image.resize(images, [self.image_size[0][0], self.image_size[0][1]])
        return images


    def _normalize(self, images):
        images = tf.cast(images, tf.float32) / 255.0
        return images

    # def _one_hot_encode(self, images, masks):
    #    """
    #    Converts mask to a one-hot encoding specified by the semantic map.
    #    """
    #    one_hot_map = []
    #    for colour in self.palette:
    #        class_map = tf.reduce_all(tf.equal(masks, colour), axis=-1)
    #        one_hot_map.append(class_map)
    #    one_hot_map = tf.stack(one_hot_map, axis=-1)
    #    one_hot_map = tf.cast(one_hot_map, tf.float32)
    #    return images, one_hot_map

    def data_all(self):
        raw_image_dataset = tf.data.TFRecordDataset(self.tf_records_path)
        for example in raw_image_dataset:
            image_features = tf.io.parse_single_example(example, self.image_feature_description)
            self.channels = [image_features['depth'], 1]
        # Parse images and labels
        data = raw_image_dataset.map(self._tf_records_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return data

    def data_batch(self, batch_size, shuffle=False, augmentation=False):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            batch_size: Number of images/masks in each batch returned.
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
        Returns:
            data: A tf dataset object.
        """

        raw_image_dataset = tf.data.TFRecordDataset(self.tf_records_path)
        for example in raw_image_dataset:
            image_features = tf.io.parse_single_example(example, self.image_feature_description)
            self.channels = [image_features['depth'], 1]
        # Parse images and labels
        data = raw_image_dataset.map(self._tf_records_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size, drop_remainder=True) # .repeat()
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data
