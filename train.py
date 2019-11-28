from src.dataset import DataLoader
from src.utils import tf_record_count
from src.model import UnetModel
import tensorflow as tf
import sys
tf.executing_eagerly()


def main(_):
    device = 'gpu:0' if tf.test.is_gpu_available else 'cpu:0'
    device = 'cpu:0'
    net = UnetModel("Unet", device, "checkpoint")

    train_dataset = DataLoader(tf_records_path="TFRecords/training.record")
    train_dataset = train_dataset.data_batch(batch_size=2, shuffle=1, augmentation=True)

    validation_dataset = DataLoader(tf_records_path="TFRecords/validation.record")
    validation_dataset = validation_dataset.data_batch(batch_size=1, shuffle=0, augmentation=False)

    optimizer = tf.optimizers.SGD()

    train_steps = tf_record_count("TFRecords/training.record")/2
    validation_steps = tf_record_count("TFRecords/validation.record")/1

    net.train(train_dataset=train_dataset, val_dataset=validation_dataset, optimizer=optimizer, train_steps=4,
              val_steps=2, plot_path='file/plot/', epochs=2)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()
