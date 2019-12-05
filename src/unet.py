import tensorflow as tf


class Conv2DLayer(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides, padding, activation, kernel_initializer):
        super(Conv2DLayer, self).__init__(name='conv_block')
        self.convolution = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                  padding=padding, activation=activation,
                                                  kernel_initializer=kernel_initializer)
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)

    def __call__(self, input_tensor, training=False):
        x = self.convolution(input_tensor)
        x = self.batch_normalization(x, training=training)
        return tf.nn.relu(x)


class UEncoder(tf.keras.Model):
    def __init__(self):
        super(UEncoder, self).__init__(name='Encoder')

        self.conv_1 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.conv_2 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv_3 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.conv_4 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv_5 = Conv2DLayer(filters=256, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.conv_6 = Conv2DLayer(filters=256, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv_7 = Conv2DLayer(filters=512, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.conv_8 = Conv2DLayer(filters=512, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        self.max_pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv_9 = Conv2DLayer(filters=1024, kernel_size=3, strides=1, padding='same',
                                  activation=None, kernel_initializer="he_normal")
        # self.conv_10 = Conv2DLayer(filters=512,  kernel_size=3, strides=1, padding='same',
        # activation=None, kernel_initializer="he_normal")
        self.skip_connections = {}

    def __call__(self, input_tensor, training):

        self.skip_connections = {0: [], 1: [], 2: [], 3: []}

        # First Downsmapling block

        x = self.conv_1(input_tensor, training=training)
        x = self.conv_2(x, training=training)
        self.skip_connections[0].append(x)
        x = self.max_pool_1(x)

        # Second Downsampling block

        x = self.conv_3(x, training=training)
        x = self.conv_4(x, training=training)
        self.skip_connections[1].append(x)
        x = self.max_pool_2(x)

        # Third Downsampling block

        x = self.conv_5(x, training=training)
        x = self.conv_6(x, training=training)
        self.skip_connections[2].append(x)
        x = self.max_pool_3(x)

        # Fourth Downsampling block

        x = self.conv_7(x, training=training)
        x = self.conv_8(x, training=training)
        self.skip_connections[3].append(x)
        x = self.max_pool_4(x)

        # Bottleneck

        x = self.conv_9(x, training=training)
        # x = self.conv_10(x, training=training)
        return x


class SkipConnection(tf.keras.Model):

    def __init__(self):
        super(SkipConnection, self).__init__(name='skip_conn')
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

    def __call__(self, input_tensor_1, input_tensor_2):
        if not isinstance(input_tensor_2, list):
            input_tensor_2 = [input_tensor_2]
        input_tensor_1 = [self.up(input_tensor_1)]
        maps = input_tensor_1 + input_tensor_2
        x = self.concatenate(maps)
        return x


class UDecoder(tf.keras.Model):

    def __init__(self):
        super(UDecoder, self).__init__(name='Decoder')

        self.conv_10 = Conv2DLayer(filters=512,  kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")
        self.up_1 = SkipConnection()

        self.conv_11 = Conv2DLayer(filters=512,  kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")
        self.conv_12 = Conv2DLayer(filters=256,  kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")
        self.up_2 = SkipConnection()

        self.conv_13 = Conv2DLayer(filters=256, kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")
        self.conv_14 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")
        self.up_3 = SkipConnection()

        self.conv_15 = Conv2DLayer(filters=128, kernel_size=3,  strides=1,  padding='same',
                                   activation=None,  kernel_initializer="he_normal")
        self.conv_16 = Conv2DLayer(filters=64,  kernel_size=3,  strides=1,  padding='same',
                                   activation=None,  kernel_initializer="he_normal")
        self.up_4 = SkipConnection()

        self.conv_17 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")
        self.conv_18 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding='same',
                                   activation=None, kernel_initializer="he_normal")

        self.final_conv = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same',
                                                 activation=None, kernel_initializer="he_normal") #, kernel_initializer="he_normal"
        # Conv2DLayer(filters=2,  kernel_size=1, strides=1, padding='same',
        # activation=None, kernel_initializer="he_normal")

    def __call__(self, input_tensor, training, skip_connections):

        # First Upsampling block

        x = self.conv_10(input_tensor, training=training)
        x = self.up_1(x, skip_connections[3])

        # Second Upsampling block

        x = self.conv_11(x, training=training)
        x = self.conv_12(x, training=training)
        x = self.up_2(x, skip_connections[2])

        # Third Upsampling block

        x = self.conv_13(x, training=training)
        x = self.conv_14(x, training=training)
        x = self.up_3(x, skip_connections[1])

        # Fourth Upsampling block

        x = self.conv_15(x, training=training)
        x = self.conv_16(x, training=training)
        x = self.up_4(x, skip_connections[0])

        # Final Block

        x = self.conv_17(x, training=training)
        x = self.conv_18(x, training=training)
        x = self.final_conv(x)
        return x
