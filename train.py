import tensorflow as tf
import sys
import tensorflow.python.eager as tfe
from src.dataset import DataLoader
from src.utils import tf_record_count
from src.model import UnetModel


def main(_):
    device = 'cpu:0'
    num_gpu = tfe.context.num_gpus()
    if num_gpu > 0:
        device = 'gpu:0'

    net = UnetModel("Unet", device, "checkpoint")

    train_batch_size = 2
    validation_batch_size = 1
    train_shuffle = 1
    train_dataset = DataLoader(tf_records_path="TFRecords/training.record")

    train_dataset = train_dataset.data_batch(batch_size=train_batch_size, shuffle=train_shuffle, augmentation=True)

    validation_dataset = DataLoader(tf_records_path="TFRecords/validation.record")
    validation_dataset = validation_dataset.data_batch(batch_size=validation_batch_size, shuffle=0, augmentation=False)

    optimizer = tf.optimizers.SGD()

    train_steps = tf_record_count("TFRecords/training.record")/train_batch_size
    validation_steps = tf_record_count("TFRecords/validation.record")/validation_batch_size

    net.train(train_dataset=train_dataset, val_dataset=validation_dataset, optimizer=optimizer, train_steps=train_steps,
              val_steps=validation_steps, plot_path='file/plot/', epochs=1)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()
