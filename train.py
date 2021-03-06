import tensorflow as tf
import sys
from src.dataset import DataLoader
from src.utils import tf_record_count
from src.model import CableModel
from src.dense_net import DenseNet


def main(_):
    device = 'cpu:0'
    print(tf.test.is_gpu_available())
    if tf.test.is_gpu_available(cuda_only=True):
        device = 'gpu:0'
    net = CableModel('Unet', device, "checkpoint")
    train_batch_size = 15
    validation_batch_size = 15
    train_dataset = DataLoader(image_size=[[128, 128], [128, 128]],
                               tf_records_path=['TFRecords/large/training_large.record'])

    train_dataset = train_dataset.train_batch(batch_size=train_batch_size, augmentation=True, shuffle=100)
    validation_dataset = DataLoader(image_size=[[128, 128], [128, 128]],
                                   tf_records_path=['TFRecords/large/validation_large.record'])
    validation_dataset = validation_dataset.validation_batch(val_batch_size=validation_batch_size)
    optimizer = tf.keras.optimizers.Adam(lr=0.0000099) # lr=0.00099
    lr = 0.000002
    train_steps = int(tf_record_count(['TFRecords/large/training_large.record'])/train_batch_size)
    validation_steps = int(tf_record_count(['TFRecords/large/validation_large.record'])/validation_batch_size)
    net.train(train_dataset=train_dataset, val_dataset=validation_dataset,
              optimizer=optimizer, train_steps=train_steps,
              val_steps=validation_steps, plot_path='file/plot/', epochs=150)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()
