import tensorflow as tf
from .unet import UEncoder, UDecoder
from .vgg16_based import Encoder, Decoder, DeepDecoder
from .metrics import precision_recall, compute_accuracy, compute_f1score
from .utils import read_pixel_frequency
from .utils import  plot, save_validation, create_folder_and_save_path, save_test, create_label_mask
from .postprocess import temp_filter
from IPython.display import clear_output
from .dilated_convolution import FrontEnd, ContextModule
import os
from PIL import Image
import numpy as np


class CableModel(tf.keras.Model):

    def __init__(self, name, device, checkpoint_dir):
        super(CableModel, self).__init__(name=name)
        self._device = device
        self._checkpoint_dir = os.path.join(checkpoint_dir, name)
        self._is_built = False
        if name == 'Unet':
            self.encoder = UEncoder()
            self.decoder = UDecoder()
        elif name == 'Vgg16':
            self.encoder = Encoder()
            self.decoder = Decoder()
        elif name == 'Vgg16NoSkip':
            self.encoder = Encoder()
            self.decoder = Decoder(skips=False)
        elif name == 'Vgg16Deep':
            self.encoder = Encoder()
            self.decoder = DeepDecoder()
        elif name == 'Vgg16DeepNoSkip':
            self.encoder = Encoder()
            self.decoder = DeepDecoder(skips=False)
        elif name == 'SimpleVgg16':
            self.encoder = FrontEnd()
            self.decoder = ContextModule('basic', 128)
        else:
            raise ValueError("UNKNOWN NET NAME")
        self.history = {}
        self.BLACK_PERCENTUAL, self.WHITE_PERCENTUAL = read_pixel_frequency('file/WHITE_BLACK_PERCENTUAL.txt')


    @property
    def device(self):
        return self._device

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def saver(self):
        return tf.compat.v1.train.Saver(self.variables)

    @property
    def is_built(self):
        return self._is_built

    @is_built.setter
    def is_built(self, value):
        if isinstance(value, bool) is False:
            raise ValueError("is_built must be a boolean value")
        self._is_built = value

    def forward(self, *args, **kwargs):
        with tf.device(self.device):
            output = self.call(*args, **kwargs)
        self.is_built = True
        return output

    def call(self, image, training=False):
        input_tensor = image
        x = self.encoder(input_tensor, training=training)
        skip_connections = self.encoder.skip_connections
        x = self.decoder(x, training=training, skip_connections=skip_connections)
        return x

    def pixel_percentual(self, one_hot_label):
        perc = 0
        count = 0
        for label in one_hot_label:
            non_zero = tf.math.count_nonzero(label, dtype=tf.float32)
            perc = float(non_zero/(label.shape[0]*label.shape[1]))
            count += 1
        perc = float(perc/count)
        return perc, 1.0 - perc

    def compute_cross_entropy(self, image, one_hot_label, training=True):
        logits = self.forward(image, training=training)
        cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=logits)
        if training:
            class_weights = tf.constant([[[[1.0/self.BLACK_PERCENTUAL, 1.0/self.WHITE_PERCENTUAL]]]])
            weights = tf.reduce_sum(class_weights * one_hot_label, axis=-1)
            weighted_loss = tf.reduce_mean(cross_entropy * weights)
            return weighted_loss
        return cross_entropy

    def backward(self, image, one_hot_label):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                loss = self.compute_cross_entropy(image, one_hot_label, training=True)
                # loss = self.class_balanced_loss(image, one_hot_label)
            gradients = tape.gradient(loss, self.trainable_variables)
            grad_vars = zip(gradients, self.trainable_variables)
            return loss, grad_vars

    def load_variables(self, sess=None, num_bands=3):
        """"
        Function to restore trained model.
        """

        if self.is_built is False:
            # Run the model once to initialize variables
            image = tf.zeros((1, 128, 128, num_bands))
            self.forward(image,  training=False)
        self.saver.restore(sess=sess, save_path=self.checkpoint_dir)

    def save_variables(self, sess=None):
        """
        Function to save trained model.
        """
        if self.is_built:
            self.saver.save(sess=sess, save_path=self.checkpoint_dir)
        else:
            raise RuntimeError("You must build the model before save model's variables.")

    def train(self, train_dataset, val_dataset, optimizer, train_steps, val_steps, plot_path, epochs=50, early_stopping=True):

        if tf.executing_eagerly() is False:
            raise RuntimeError("train method must be run only with eager execution.")
        global_step = tf.compat.v1.train.get_or_create_global_step()
        best_f1score = 0
        last_improvement = 0
        # Initialize dictionary to store the history
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [], 'val_f1score': [], 'train_acc': [],
                        'val_acc': []}
        save_path = create_folder_and_save_path('file/training_predictions/',  self.name, split=False)
        for i in range(epochs):
            epoca = i
            clear_output()
            print("Epoch: {}/{}".format(i + 1, epochs))

            train_loss = tf.metrics.Mean('train_loss')
            train_f1score = tf.metrics.Mean('train_f1')
            train_accuracy = tf.metrics.Mean('train_acc')

            train_progbar = tf.keras.utils.Progbar(train_steps)

            for b, (image, mask) in enumerate(train_dataset):

                mask = tf.squeeze(mask, axis=-1)
                one_hot_labels = tf.one_hot(indices=mask, depth=2, dtype=tf.float32)
                mask = tf.cast(mask, tf.float32)
                #self.WHITE_PERCENTUAL, self.BLACK_PERCENTUAL = self.pixel_percentual(mask)
                loss, grads = self.backward(image, one_hot_labels)
                optimizer.apply_gradients(grads, global_step)
                logits = self.forward(image, training=False)
                precision, recall = precision_recall(logits, mask)
                f1score = compute_f1score(precision, recall)
                accuracy = compute_accuracy(logits, mask)
                train_loss(loss)
                train_f1score(f1score)
                train_accuracy(accuracy)
                metrics = [('loss', loss), ("f1", f1score), ("accuracy", accuracy)]
                train_progbar.update(b + 1, metrics)

            val_loss = tf.metrics.Mean('val_loss')
            val_f1score = tf.metrics.Mean('val_f1')
            val_accuracy = tf.metrics.Mean('val_acc')

            val_progbar = tf.keras.utils.Progbar(val_steps)
            print("\nVALIDATION")
            for b, (image, mask) in enumerate(val_dataset):

                mask = tf.squeeze(mask, axis=-1)
                one_hot_labels = tf.one_hot(indices=mask, depth=2, dtype=tf.float32)
                mask = tf.cast(mask, tf.float32)

                loss = self.compute_cross_entropy(image, one_hot_labels, training=False)
                # loss = self.class_balanced_loss(image, one_hot_labels, training=False)
                logits = self.forward(image, training=False)

                precision, recall = precision_recall(logits, mask)
                f1score = compute_f1score(precision, recall)
                accuracy = compute_accuracy(logits, mask)
                val_loss(loss)
                val_f1score(f1score)
                val_accuracy(accuracy)
                metrics = [('val_loss', loss), ("val_f1", f1score), ("val_acc", accuracy)]
                val_progbar.update(b + 1, metrics)

            for b, (image, mask) in enumerate(val_dataset):
                if b >= 3:
                    break
                logits = self.forward(image, training=False)
                save_validation(image[0], mask[0], logits[0], save_path, '_epoch_' + str(epoca + 1) + '_' + str(b))

            self.history['train_loss'].append(train_loss.result().numpy())
            self.history['train_acc'].append(train_accuracy.result().numpy())
            self.history['train_f1score'].append(train_f1score.result().numpy())

            self.history['val_loss'].append(val_loss.result().numpy())
            self.history['val_acc'].append(val_accuracy.result().numpy())
            self.history['val_f1score'].append(val_f1score.result().numpy())

            if self.history['val_f1score'][-1] >= best_f1score:
                last_improvement = 0
                self.save_variables()
                print("Model saved. f1score : {} --> {}".format(best_f1score, self.history['val_f1score'][-1]))
                best_f1score = self.history['val_f1score'][-1]
            else :
                last_improvement += 1
            if last_improvement >= 15:
                break
            if i != 0 and i%10 == 0:
                plot(plot_path, self.name, self.history, i + 1)
        plot(plot_path, self.name, self.history, i + 1)


    def test(self, test_dataset, steps):
        if tf.executing_eagerly() is False:
            raise RuntimeError("evaluate method must be run only with eager execution.")
        self.load_variables()
        progbar = tf.keras.utils.Progbar(steps)
        save_path = create_folder_and_save_path('file/test/', self.name, split=False)
        count = 0
        for b, (image) in enumerate(test_dataset):
            logits = self.forward(image, training=False)
            progbar.update(b + 1)
            for logit in logits:
                lgt = create_label_mask(logit)
                prediction = (tf.keras.preprocessing.image.array_to_img(lgt))
                prediction.save(os.path.join(save_path,"test_"+str(count)+'.png'))
                count = count + 1

    def test_with_filter(self, test_dataset, steps):
        window = 30
        batch = []
        if tf.executing_eagerly() is False:
            raise RuntimeError("evaluate method must be run only with eager execution.")
        self.load_variables()
        progbar = tf.keras.utils.Progbar(steps)
        save_path = create_folder_and_save_path('file/test/', self.name, split=False)
        count = 0
        for b, (image) in enumerate(test_dataset):
            logits = self.forward(image, training=False)
            progbar.update(b + 1)
            for logit in logits:
                lgt = create_label_mask(logit)
                if len(batch) == window :
                    mean = temp_filter(batch)
                    filtered = (tf.keras.preprocessing.image.array_to_img(mean))
                    filtered.save(os.path.join(save_path, "filtered_"+str(count)+'.png'))
                    count = count + 1
                    batch = batch[1:]

    def evaluate(self, test_dataset, steps):
        if tf.executing_eagerly() is False:
            raise RuntimeError("evaluate method must be run only with eager execution.")
        self.load_variables()
        precision_mean = tf.metrics.Mean('precision')
        recall_mean = tf.metrics.Mean('recall')
        accuracy_mean = tf.metrics.Mean('accuracy')
        f1score_mean = tf.metrics.Mean('f1score')
        progbar = tf.keras.utils.Progbar(steps)
        save_path = create_folder_and_save_path('file/predictions/', self.name, split=False)
        count = 0
        for b, (image, mask) in enumerate(test_dataset):
            labels = tf.cast(tf.squeeze(mask, axis=-1), tf.float32)
            logits = self.forward(image, training=False)

            precision, recall = precision_recall(logits, labels)
            accuracy = compute_accuracy(logits, labels)
            f1score = compute_f1score(precision, recall)
            count += 1
            if count < 50:
                for i in range(0, int(len(image) / 10)):
                    save_validation(image[i], mask[i], logits[i], save_path, count)
            precision_mean(precision)
            recall_mean(recall)
            accuracy_mean(accuracy)
            f1score_mean(f1score)
            progbar.update(b + 1)

        print("EVALUATE {}".format(self.name))
        print("precision: {}".format(precision_mean.result().numpy()))
        print("recall: {}".format(recall_mean.result().numpy()))
        print("accuracy: {}".format(accuracy_mean.result().numpy()))
        print("f1score: {}".format(f1score_mean.result().numpy()))
