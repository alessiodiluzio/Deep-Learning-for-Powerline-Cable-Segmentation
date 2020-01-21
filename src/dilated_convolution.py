import tensorflow as tf


class Conv2DLayer(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides, padding, activation, kernel_initializer, dilation_rate=1):
        super(Conv2DLayer, self).__init__(name='conv_block')
        self.convolution = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                  padding=padding, activation=activation,
                                                  kernel_initializer=kernel_initializer,dilation_rate=dilation_rate)
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)

    def __call__(self, input_tensor, training=False):
        x = self.convolution(input_tensor)
        x = self.batch_normalization(x, training=training)
        return tf.nn.relu(x)


class FrontEnd(tf.keras.Model):
    def __init__(self):
        super(FrontEnd, self).__init__(name='FrontEnd')
        base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                                       input_shape=(128, 128, 3), pooling=None, classes=2)
        base_model.trainable = False
        self.skip_connections = []
        layer_names = [
            'block1_conv1',  # 64x64
            'block1_conv2',  # 32x32
            'block1_pool',  # 16x16
            'block2_conv1',  # 8x8
            'block2_conv2',  # 4x4
        ]
        layers = [base_model.get_layer(name) for name in layer_names]
        for layer in layers:
            layer.output.trainable = False
        self.conv_1 = layers[0]
        self.conv_2 = layers[1]
        self.conv_3 = layers[3]
        self.conv_4 = layers[4]
        """self.conv_1 = Conv2DLayer(filters=32,kernel_size=3, padding='same',strides=1,
                                  activation=None, kernel_initializer="he_normal")
        self.conv_1.set_weights(layers[0].get_weights())
        self.conv_2 = Conv2DLayer(filters=32,kernel_size=3, padding='same',strides=1,
                                  activation=None, kernel_initializer="he_normal")
        self.conv_2.set_weights(layers[1].get_weights())
        self.conv_3 = Conv2DLayer(filters=32,kernel_size=3, padding='same',strides=1,
                                  activation=None, kernel_initializer="he_normal")
        self.conv_3.set_weights(layers[3].get_weights())
        self.conv_4 = Conv2DLayer(filters=32,kernel_size=3, padding='same',strides=1,
                                  activation=None, kernel_initializer="he_normal")
        self.conv_4.set_weights(layers[4].get_weights())"""


    def __call__(self, input_tensor, training):

        x = self.conv_1(input_tensor, training=training)
        x = self.conv_2(x, training=training)
        x = self.conv_3(x, training=training)
        x = self.conv_4(x, training=training)

        return x


class ContextModule(tf.keras.Model):

    def __init__(self, output_channels, input_maps):
        super(ContextModule, self).__init__(name='ContextModule')
        C = input_maps
        if output_channels == 'basic':
            filters = [32 for i in range(8)]
        elif output_channels == 'large':
            filters = [2*C, 2*C, 4*C, 8*C, 16*C, 32*C, 32*C, C]

        self.dil_conv_1 = Conv2DLayer(filters=filters[0], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=1)
        self.dil_conv_2 = Conv2DLayer(filters=filters[1], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=1)
        self.dil_conv_3 = Conv2DLayer(filters=filters[2], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=2)
        self.dil_conv_4 = Conv2DLayer(filters=filters[3], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=4)
        self.dil_conv_5 = Conv2DLayer(filters=filters[4], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=8)
        self.dil_conv_6 = Conv2DLayer(filters=filters[5], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=16)
        self.dil_conv_7 = Conv2DLayer(filters=filters[6], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=1)
        self.dil_conv_8 = Conv2DLayer(filters=filters[7], kernel_size=3, strides=1, padding="same", activation='relu',
                                            kernel_initializer='he_normal', dilation_rate=1)
        self.last_conv = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same', activation=None,
                                                 kernel_initializer="he_normal")

    def __call__(self, input_tensor, training, skip_connections):

        x = self.dil_conv_1(input_tensor, training=training)
        x = self.dil_conv_2(x, training=training)
        x = self.dil_conv_3(x, training=training)
        x = self.dil_conv_4(x, training=training)
        x = self.dil_conv_5(x, training=training)
        x = self.dil_conv_6(x, training=training)
        x = self.dil_conv_7(x, training=training)
        x = self.dil_conv_8(x, training=training)
        x = self.last_conv(x)
        return x
