import tensorflow as tf
import random
from src.utils import display_image, create_label_mask


class DataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems."""

    def __init__(self, tf_records_path, palette=(0, 1)):
        """
        Initializes the data loader object
        Args:
            tf_records_path: Path to the TFRecord file for images and masks dataset
            palette: A list of RGB pixel values in the mask. If specified, the mask
                     will be one hot encoded along the channel dimension.
        """
        self.tf_records_path = tf_records_path
        self.palette = palette
        self.image_size = []
        self.channels = []
        self.image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'mask_raw': tf.io.FixedLenFeature([], tf.string),

        }

    def _tf_records_parser(self, example):

        image_features = tf.io.parse_single_example(example, self.image_feature_description)
        image_raw = image_features['image_raw']
        images = tf.io.decode_png(image_raw, channels=self.channels[0])
        mask_raw = image_features['mask_raw']
        masks = tf.io.decode_png(mask_raw, channels=self.channels[1])
        images, masks = self._resize_data(images, masks)
        images, masks = self._normalize(images, masks)
        return images, masks

    def _resize_data(self, images, masks):
        """
        Resizes images to specified size.
        """
        images = tf.image.resize(images, [self.image_size[0][0], self.image_size[0][1]])
        masks = tf.image.resize(masks, [self.image_size[1][0], self.image_size[1][1]], method='nearest')

        return images, masks

    @staticmethod
    def _flip(images, masks):
        seed = random.random()
        images = tf.image.random_flip_left_right(images, seed=seed)
        masks = tf.image.random_flip_left_right(masks, seed=seed)
        return images, masks

    @staticmethod
    def _brightness(images, masks):
        seed = random.random()
        # max_delta = float(random.randrange(1, 4)) / 10
        images = tf.image.random_brightness(images, max_delta=0.2, seed=seed)
        return images, masks

    @staticmethod
    def _saturation(images, masks):
        seed = random.random()
        # upper = float(random.randrange(1, 4)) / 10
        images = tf.image.random_saturation(images, lower=1, upper=2.5, seed=seed)
        return images, masks

    @staticmethod
    def _contrast(images, masks):
        seed = random.random()
        # upper = float(random.randrange(1, 4)) / 10
        images = tf.image.random_contrast(images, lower=1, upper=2.5, seed=seed)
        return images, masks

    @staticmethod
    def _normalize(images, masks):
        images = tf.cast(images, tf.float32) / 255.0
        masks = tf.cast(masks, tf.float32) / 255.0
        return images, masks

    def _one_hot_encode(self, images, masks):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        one_hot_map = []
        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(masks, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        return images, one_hot_map

    def data_batch(self, batch_size, one_hot_encode=True, shuffle=10, augmentation=False):
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
            self.channels = [image_features['depth'], image_features['depth']]
            self.image_size = [[image_features['height'], image_features['width']],
                               [image_features['height'], image_features['width']]]
        # Parse images and labels
        data = raw_image_dataset.map(self._tf_records_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if augmentation:
            data = data.map(self._flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            data = data.map(self._contrast, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            data = data.map(self._brightness, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            data = data.map(self._saturation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if one_hot_encode:
            if self.palette is None:
                raise ValueError('No Palette for one-hot encoding specified in the data loader! \
                                  please specify one when initializing the loader.')
            data = data.map(self._one_hot_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle:
            data = data.shuffle(shuffle)
        data = data.batch(batch_size).repeat()
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data


train_dataset = DataLoader(tf_records_path="TFRecords/images.tfrecords")
train_dataset = train_dataset.data_batch(batch_size=100, shuffle=10, augmentation=True)

for image, mask in train_dataset.take(3):
    display_image([image[0], create_label_mask(mask[0])])
"""""
validation_dataset = DataLoader(image_paths=test_images,
                                mask_paths=test_images_label,
                                image_sizes=[[256, 256], [256, 256]],
                                palette=[0, 1])

validation_dataset = validation_dataset.data_batch(batch_size=1)
"""""
