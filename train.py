import tensorflow as tf
import sys
from src.dataset import DataLoader, TestDataLoader
from src.utils import tf_record_count
from src.model import CableModel


def main(_):
    device = 'cpu:0'
    print(tf.test.is_gpu_available())
    if tf.test.is_gpu_available(cuda_only=True):
        device = 'gpu:0'
    net = CableModel('Unet', device, "checkpoint")
    train_batch_size = 10
    validation_batch_size = 10
    train_dataset = DataLoader(image_size=[[128, 128], [128, 128]],
                              tf_records_path=['TFRecords/small/training_small.record'])
    train_dataset = train_dataset.data_batch(batch_size=train_batch_size, augmentation=True, shuffle=100)
    validation_dataset = DataLoader(image_size=[[128, 128], [128, 128]],
                                    tf_records_path=['TFRecords/small/validation_small.record'])
    validation_dataset = validation_dataset.data_batch(batch_size=validation_batch_size, augmentation=False)
    optimizer = tf.keras.optimizers.Adam(lr=0.0000099) # lr = 0.00005
    # lr = 0.000002
    train_steps = int(tf_record_count(['TFRecords/small/training_small.record'])/train_batch_size)
    validation_steps = int(tf_record_count(['TFRecords/small/validation_small.record'])/validation_batch_size)
    net.train(train_dataset=train_dataset, val_dataset=validation_dataset, optimizer=optimizer, train_steps=train_steps,
            val_steps=validation_steps, plot_path='file/plot/', epochs=50)

    net.evaluate(validation_dataset, validation_steps)
    test_batch_size = 10

    test_dataset = TestDataLoader(image_size=[[128, 128], [128, 128]],
                                  tf_records_path=['TFRecords/real/real.record'])
    test_dataset = test_dataset.data_batch(batch_size=test_batch_size, augmentation=False)

    test_steps = int(tf_record_count(['TFRecords/real/real.record'])/test_batch_size)
    net.test(test_dataset, test_steps)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()
