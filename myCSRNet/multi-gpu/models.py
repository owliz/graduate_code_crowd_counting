import tensorflow as tf


class Model(object):
    """Fully convolutional autoencoder for temporal-regularity detection.
        For simplicity, fully convolutional autoencoder structure is
        changed to be fixed as symmetric.

        Reference:
        [1] Learning Temporal Regularity in Video Sequences
            (http://arxiv.org/abs/1604.04574)
        [2] https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
    """

    def preprocess_input(x):
        """Preprocesses a tensor encoding a batch of images.

        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.

        # Returns
            Preprocessed tensor.
        """

        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x = x - tf.stack((tf.ones_like(x[:, :, :, 0]) * tf.constant(103.939),
                          tf.ones_like(x[:, :, :, 1]) * tf.constant(116.779)
                          , tf.ones_like(x[:, :, :, 2]) * tf.constant(123.68)), axis=-1)

        # x = 2*x/255
        return x

    def __init__(self, cfg, images, batch_size, type='a'):
        """
        Args:
          sess : TensorFlow session
          input_shape : Shape of the input data. [batch_size, clip_length, h, w, 1]
        """

        self._x = self.preprocess_input(images)
        self._batch_size = batch_size
        self._h = cfg.height
        self._w = cfg.width
        self._c = self._x.get_shape().as_list()[-1]
        self._build_front()
        self._build_backend()

    def _build_front(self):



    def _build_backend(self):
        # with tf.variable_scope("generator"):
        # TimeDistributed 就是将 Conv2D 分别运用到t帧上

        # CONV1
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,256,256,3) -> (batch_size,time_length,256,256,16)
        distributed_conv2d_1 = self._distributedConv2D(self._x, output_channels=16,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_1')
        distributed_conv2d_1 = self._lrelu(distributed_conv2d_1)
        distributed_conv2d_1 = tf.layers.batch_normalization(distributed_conv2d_1)

        # CONV2
        # filter/stride: (5,5)/(2,2)
        # (batch_size,time_length,256,256,16) -> (batch_size,time_length,128,128,16)
        distributed_conv2d_2 = self._distributedConv2D(distributed_conv2d_1, output_channels=16,
                                                       k_h=5,
                                                       k_w=5, s_h=2, s_w=2,
                                                       name='distributed_conv2d_2')
        distributed_conv2d_2 = self._lrelu(distributed_conv2d_2)
        distributed_conv2d_2 = tf.layers.batch_normalization(distributed_conv2d_2)

        # CONV3
        # filter/stride: (5,5)/(2,2)
        # (batch_size,time_length,128,128,16) -> (batch_size,time_length,64,64,32)
        distributed_conv2d_3 = self._distributedConv2D(distributed_conv2d_2, output_channels=32,
                                                       k_h=5,
                                                       k_w=5, s_h=2, s_w=2,
                                                       name='distributed_conv2d_3')
        distributed_conv2d_3 = self._lrelu(distributed_conv2d_3)
        distributed_conv2d_3 = tf.layers.batch_normalization(distributed_conv2d_3)

        # CONV4
        # filter/stride: (3,3)/(2,2)
        # (batch_size,time_length,64,64,32) -> (batch_size,time_length,32,32,64)
        distributed_conv2d_4 = self._distributedConv2D(distributed_conv2d_3, output_channels=64,
                                                       k_h=3,
                                                       k_w=3, s_h=2, s_w=2,
                                                       name='distributed_conv2d_4')
        distributed_conv2d_4 = self._lrelu(distributed_conv2d_4)
        distributed_conv2d_4 = tf.layers.batch_normalization(distributed_conv2d_4)

        # CONV5
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,32,32,64) -> (batch_size,time_length,32,32,128)
        distributed_conv2d_5 = self._distributedConv2D(distributed_conv2d_4, output_channels=128,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_5')
        distributed_conv2d_5 = self._lrelu(distributed_conv2d_5)
        distributed_conv2d_5 = tf.layers.batch_normalization(distributed_conv2d_5)

        # CONVLSTM
        # temporal encoder-decoder
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,32,32,128) -> (batch_size,time_length,32,32,128)
        convlstm1 = self._convlstm(distributed_conv2d_5, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_1")
        convlstm1 = tf.layers.batch_normalization(convlstm1)
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,32,32,128) -> (batch_size,time_length,32,32,64)
        convlstm2 = self._convlstm(convlstm1, output_channels=64, kernel_size=[3, 3],
                                   name="conv_lstm_cell_2")

        """
        predict stream
        """
        with tf.variable_scope('predict'):
            # filter/stride: (3,3)/(1,1)
            # (batch_size,time_length,32,32,64) -> (batch_size,time_length,32,32,128)
            convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
                                       name="conv_lstm_cell_3")
            # (batch_size,1,32,32,128)
            convlstm3_final = convlstm3[:, -1:-2:-1, ...]

            # DECONV1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,32,32,128) -> (batch_size,1,32,32,64)
            deconv2d_1 = self._distributedDeConv2D(convlstm3_final,
                                                   output_channels=64,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       32, 32, 64],
                                                   k_h=3, k_w=3, s_h=1, s_w=1,
                                                   name='deconv2d_1')
            deconv2d_1 = self._lrelu(deconv2d_1)
            deconv2d_1 = tf.layers.batch_normalization(deconv2d_1)
            # deconv2d_1 = tf.nn.dropout(deconv2d_1, 0.5)

            # DECONV2
            # filter/stride: (3,3)/(2,2)
            # (batch_size,1,32,32,64) -> (batch_size,1,64,64,32)
            deconv2d_2 = self._distributedDeConv2D(deconv2d_1,
                                                   output_channels=32,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       64, 64, 32],
                                                   k_h=3, k_w=3, s_h=2, s_w=2,
                                                   name='deconv2d_2')
            deconv2d_2 = self._lrelu(deconv2d_2)
            deconv2d_2 = tf.layers.batch_normalization(deconv2d_2)
            # deconv2d_2 = tf.nn.dropout(deconv2d_2, 0.5)

            # DECONV3
            # filter/stride: (5,5)/(2,2)
            # (batch_size,1,64,64,32) -> (batch_size,1,128,128,16)
            deconv2d_3 = self._distributedDeConv2D(deconv2d_2,
                                                   output_channels=16,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       128, 128, 16],
                                                   k_h=5, k_w=5, s_h=2, s_w=2,
                                                   name='deconv2d_3')
            deconv2d_3 = self._lrelu(deconv2d_3)
            deconv2d_3 = tf.layers.batch_normalization(deconv2d_3)
            # deconv2d_3 = tf.nn.dropout(deconv2d_3, 0.5)

            # DECONV4
            # filter/stride: (5,5)/(2,2)
            # (batch_size,1,128,128,16) -> (batch_size,1,256,256,16)
            deconv2d_4 = self._distributedDeConv2D(deconv2d_3,
                                                   output_channels=16,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       256, 256, 16],
                                                   k_h=5, k_w=5, s_h=2, s_w=2,
                                                   name='deconv2d_4')
            deconv2d_4 = self._lrelu(deconv2d_4)
            deconv2d_4 = tf.layers.batch_normalization(deconv2d_4)

            # DECONV5
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,256,256,16) -> (batch_size,1,256,256,input_channel)
            deconv2d_5 = self._distributedDeConv2D(deconv2d_4,
                                                   output_channels=self._c,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       256, 256, self._c],
                                                   k_h=3, k_w=3, s_h=1, s_w=1,
                                                   name='deconv2d_5')
            deconv2d_5 = tf.nn.tanh(deconv2d_5)

            # [batch_size,256,256,input_channel]
            self._pred_out = tf.squeeze(deconv2d_5, axis=[1])

        """
        reconstruct stream
        """
        with tf.variable_scope('reconstruct'):
            # (batch_size,1,32,32,128) -> (batch_size,1,32,32,128)
            convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
                                       name="conv_lstm_cell_3")
            # DECONV1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,32,32,128) -> (batch_size,1,32,32,64)
            distributed_deconv2d_1 = self._distributedDeConv2D(convlstm3, output_channels=64,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   32, 32, 64],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='distributed_deconv2d_1')
            distributed_deconv2d_1 = self._lrelu(distributed_deconv2d_1)
            distributed_deconv2d_1 = tf.layers.batch_normalization(distributed_deconv2d_1)

            # DECONV2
            # filter/stride: (3,3)/(2,2)
            # (batch_size,1,32,32,128) -> (batch_size,1,64,64,32)
            distributed_deconv2d_2 = self._distributedDeConv2D(distributed_deconv2d_1,
                                                               output_channels=32,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   64, 64, 32],
                                                               k_h=3, k_w=3, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_2')
            distributed_deconv2d_2 = self._lrelu(distributed_deconv2d_2)
            distributed_deconv2d_2 = tf.layers.batch_normalization(distributed_deconv2d_2)

            # DECONV3
            # filter/stride: (5,5)/(2,2)
            # (batch_size,1,64,64,32) -> (batch_size,1,128,128,16)
            distributed_deconv2d_3 = self._distributedDeConv2D(distributed_deconv2d_2,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   128, 128, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_3')
            distributed_deconv2d_3 = self._lrelu(distributed_deconv2d_3)
            distributed_deconv2d_3 = tf.layers.batch_normalization(distributed_deconv2d_3)

            # DECONV4
            # filter/stride: (5,2)/(5,2)
            # (batch_size,1,128,128,16) -> (batch_size,1,256,256,16)
            distributed_deconv2d_4 = self._distributedDeConv2D(distributed_deconv2d_3,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   256, 256, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_4')
            distributed_deconv2d_4 = self._lrelu(distributed_deconv2d_4)
            distributed_deconv2d_4 = tf.layers.batch_normalization(distributed_deconv2d_4)

            # DECONV5
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,256,256,16) -> (batch_size,1,256,256,input_c)
            distributed_deconv2d_5 = self._distributedDeConv2D(distributed_deconv2d_4,
                                                               output_channels=self._c,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   256, 256, self._c],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='distributed_deconv2d_5')
            # distributed_deconv2d_5 = self._lrelu(distributed_deconv2d_5)
            # distributed_deconv2d_5 = tf.layers.batch_normalization(distributed_deconv2d_5)
            # [self._batch_size, time_length, 256, 256, input_c]
            self._reconstr_out = distributed_deconv2d_5


    def _skip_build(self):
        # with tf.variable_scope("generator"):
        # TimeDistributed 就是将 Conv2D 分别运用到t帧上
        conv = []
        # CONV1
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,256,256,3) -> (batch_size,time_length,256,256,16)
        distributed_conv2d_1 = self._distributedConv2D(self._x, output_channels=16,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_1')
        distributed_conv2d_1 = self._lrelu(distributed_conv2d_1)
        distributed_conv2d_1 = tf.layers.batch_normalization(distributed_conv2d_1)

        # CONV1_1
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,256,256,16) -> (batch_size,time_length,256,256,16)
        distributed_conv2d_1_1 = self._distributedConv2D(distributed_conv2d_1,
                                                         output_channels=16,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_1_1')
        distributed_conv2d_1_1 = self._lrelu(distributed_conv2d_1_1)
        distributed_conv2d_1_1 = tf.layers.batch_normalization(distributed_conv2d_1_1)

        # conv[0]: (batch_size,1,256,256,16)
        conv.append(distributed_conv2d_1_1[:, -1:-2:-1, ...])

        # CONV2
        # filter/stride: (5,5)/(2,2)
        # (batch_size,time_length,256,256,16) -> (batch_size,time_length,128,128,16)
        distributed_conv2d_2 = self._distributedConv2D(distributed_conv2d_1_1, output_channels=16,
                                                       k_h=5,
                                                       k_w=5, s_h=2, s_w=2,
                                                       name='distributed_conv2d_2')
        distributed_conv2d_2 = self._lrelu(distributed_conv2d_2)
        distributed_conv2d_2 = tf.layers.batch_normalization(distributed_conv2d_2)
        # CONV2_1
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,128,128,16) -> (batch_size,time_length,128,128,16)
        distributed_conv2d_2_1 = self._distributedConv2D(distributed_conv2d_2, output_channels=16,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_2_1')
        distributed_conv2d_2_1 = self._lrelu(distributed_conv2d_2_1)
        distributed_conv2d_2_1 = tf.layers.batch_normalization(distributed_conv2d_2_1)

        # conv[1]: (batch_size,1,128,128,16)
        conv.append(distributed_conv2d_2_1[:, -1:-2:-1, ...])

        # CONV3
        # filter/stride: (5,5)/(2,2)
        # (batch_size,time_length,128,128,16) -> (batch_size,time_length,64,64,32)
        distributed_conv2d_3 = self._distributedConv2D(distributed_conv2d_2_1, output_channels=32,
                                                       k_h=5,
                                                       k_w=5, s_h=2, s_w=2,
                                                       name='distributed_conv2d_3')
        distributed_conv2d_3 = self._lrelu(distributed_conv2d_3)
        distributed_conv2d_3 = tf.layers.batch_normalization(distributed_conv2d_3)

        # CONV3_1
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,64,64,32) -> (batch_size,time_length,64,64,32)
        distributed_conv2d_3_1 = self._distributedConv2D(distributed_conv2d_3, output_channels=32,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_3_1')
        distributed_conv2d_3_1 = self._lrelu(distributed_conv2d_3_1)
        distributed_conv2d_3_1 = tf.layers.batch_normalization(distributed_conv2d_3_1)

        # conv[2]: (batch_size,1,64,64,32)
        conv.append(distributed_conv2d_3_1[:, -1:-2:-1, ...])
        # CONV4
        # filter/stride: (3,3)/(2,2)
        # (batch_size,time_length,64,64,32) -> (batch_size,time_length,32,32,64)
        distributed_conv2d_4 = self._distributedConv2D(distributed_conv2d_3, output_channels=64,
                                                       k_h=3,
                                                       k_w=3, s_h=2, s_w=2,
                                                       name='distributed_conv2d_4')
        distributed_conv2d_4 = self._lrelu(distributed_conv2d_4)
        distributed_conv2d_4 = tf.layers.batch_normalization(distributed_conv2d_4)
        
        # CONV5
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,32,32,64) -> (batch_size,time_length,32,32,128)
        distributed_conv2d_5 = self._distributedConv2D(distributed_conv2d_4, output_channels=128,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_5')
        distributed_conv2d_5 = self._lrelu(distributed_conv2d_5)
        distributed_conv2d_5 = tf.layers.batch_normalization(distributed_conv2d_5)

        # CONVLSTM
        # temporal encoder-decoder
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,32,32,128) -> (batch_size,time_length,32,32,128)
        convlstm1 = self._convlstm(distributed_conv2d_5, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_1")
        convlstm1 = tf.layers.batch_normalization(convlstm1)
        # filter/stride: (3,3)/(1,1)
        # (batch_size,time_length,32,32,128) -> (batch_size,time_length,32,32,64)
        convlstm2 = self._convlstm(convlstm1, output_channels=64, kernel_size=[3, 3],
                                   name="conv_lstm_cell_2")

        """
        predict stream
        """
        with tf.variable_scope('predict'):
            # filter/stride: (3,3)/(1,1)
            # (batch_size,time_length,32,32,64) -> (batch_size,time_length,32,32,128)
            convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_3")
            # (batch_size,1,32,32,128)
            convlstm3_final = convlstm3[:, -1:-2:-1, ...]
            
            # DECONV1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,32,32,128) -> (batch_size,1,32,32,64)
            deconv2d_1 = self._distributedDeConv2D(convlstm3_final, 
                                                               output_channels=64,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   32, 32, 64],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='deconv2d_1')
            deconv2d_1 = self._lrelu(deconv2d_1)
            deconv2d_1 = tf.layers.batch_normalization(deconv2d_1)
            # deconv2d_1 = tf.nn.dropout(deconv2d_1, 0.5)

            # DECONV2
            # filter/stride: (3,3)/(2,2)
            # (batch_size,1,32,32,64) -> (batch_size,1,64,64,32)
            deconv2d_2 = self._distributedDeConv2D(deconv2d_1,
                                                               output_channels=32,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   64, 64, 32],
                                                               k_h=3, k_w=3, s_h=2, s_w=2,
                                                               name='deconv2d_2')
            deconv2d_2 = self._lrelu(deconv2d_2)
            deconv2d_2 = tf.layers.batch_normalization(deconv2d_2)
            # deconv2d_2 = tf.nn.dropout(deconv2d_2, 0.5)

            # DECONV2_1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,64,64,32) -> (batch_size,1,64,64,32)
            deconv2d_2_1 = self._distributedDeConv2D(deconv2d_2,
                                                   output_channels=32,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       64, 64, 32],
                                                   k_h=3, k_w=3, s_h=1, s_w=1,
                                                   name='deconv2d_2_1')
            deconv2d_2_1 = self._lrelu(deconv2d_2_1)
            deconv2d_2_1 = tf.layers.batch_normalization(deconv2d_2_1)

            # CONCAT_CONV1
            # (batch_size,1,64,64,32) -> (batch_size,1,64,64,64)
            concat_conv_1 = tf.concat([conv[2], deconv2d_2_1], axis=4)
            # (batch_size,1,64,64,64) -> (batch_size,1,64,64,32)
            concat_conv_1 = self._distributedConv2D(concat_conv_1, output_channels=32,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='concat_conv_1')
            concat_conv_1 = self._lrelu(concat_conv_1)
            concat_conv_1 = tf.layers.batch_normalization(concat_conv_1)

            # DECONV3
            # filter/stride: (5,5)/(2,2)
            # (batch_size,1,64,64,32) -> (batch_size,1,128,128,16)
            deconv2d_3 = self._distributedDeConv2D(concat_conv_1,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   128, 128, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='deconv2d_3')
            deconv2d_3 = self._lrelu(deconv2d_3)
            deconv2d_3 = tf.layers.batch_normalization(deconv2d_3)

            # DECONV3_1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,128,128,16) -> (batch_size,1,128,128,16)
            deconv2d_3_1 = self._distributedDeConv2D(deconv2d_3,
                                                   output_channels=16,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       128, 128, 16],
                                                   k_h=3, k_w=3, s_h=1, s_w=1,
                                                   name='deconv2d_3_1')
            deconv2d_3_1 = self._lrelu(deconv2d_3_1)
            deconv2d_3_1 = tf.layers.batch_normalization(deconv2d_3_1)

            # deconv2d_3 = tf.nn.dropout(deconv2d_3, 0.5)
            # CONCAT_CONV2
            # (batch_size,1,128,128,16) -> (batch_size,1,128,128,32)
            concat_conv_2 = tf.concat([conv[1], deconv2d_3_1], axis=4)
            # (batch_size,1,128,128,32) -> (batch_size,1,128,128,16)
            concat_conv_2 = self._distributedConv2D(concat_conv_2, output_channels=16,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='concat_conv_2')
            concat_conv_2 = self._lrelu(concat_conv_2)
            concat_conv_2 = tf.layers.batch_normalization(concat_conv_2)

            # DECONV4
            # filter/stride: (5,5)/(2,2)
            # (batch_size,1,128,128,16) -> (batch_size,1,256,256,16)
            deconv2d_4 = self._distributedDeConv2D(concat_conv_2,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   256, 256, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='deconv2d_4')
            deconv2d_4 = self._lrelu(deconv2d_4)
            deconv2d_4 = tf.layers.batch_normalization(deconv2d_4)

            # DECONV4_1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,256,256,16) -> (batch_size,1,256,256,16)
            deconv2d_4_1 = self._distributedDeConv2D(deconv2d_4,
                                                   output_channels=16,
                                                   flatten_output_shape=[
                                                       self._batch_size,
                                                       256, 256, 16],
                                                   k_h=3, k_w=3, s_h=1, s_w=1,
                                                   name='deconv2d_4_1')
            deconv2d_4_1 = self._lrelu(deconv2d_4_1)
            deconv2d_4_1 = tf.layers.batch_normalization(deconv2d_4_1)

            # CONCAT_CONV3
            # (batch_size,1,256,256,16) -> (batch_size,1,256,256,32)
            concat_conv_3 = tf.concat([conv[0], deconv2d_4_1], axis=4)
            # (batch_size,1,256,256,32) -> (batch_size,1,256,256,16)
            concat_conv_3 = self._distributedConv2D(concat_conv_3, output_channels=16,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='concat_conv_3')
            concat_conv_3 = self._lrelu(concat_conv_3)
            concat_conv_3 = tf.layers.batch_normalization(concat_conv_3)

            # DECONV5
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,256,256,16) -> (batch_size,1,256,256,input_channel)
            deconv2d_5 = self._distributedDeConv2D(concat_conv_3, 
                                                               output_channels=self._c,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   256, 256, self._c],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='deconv2d_5')
            deconv2d_5 = tf.nn.tanh(deconv2d_5)

            # [batch_size,128,128,input_channel]
            self._pred_out = tf.squeeze(deconv2d_5, axis=[1])

        """
        reconstruct stream
        """
        with tf.variable_scope('reconstruct'):
            # (batch_size,1,32,32,128) -> (batch_size,1,32,32,128)
            convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_3")
            # DECONV1
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,32,32,128) -> (batch_size,1,32,32,64)
            distributed_deconv2d_1 = self._distributedDeConv2D(convlstm3, output_channels=64,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   32, 32, 64],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='distributed_deconv2d_1')
            distributed_deconv2d_1 = self._lrelu(distributed_deconv2d_1)
            distributed_deconv2d_1 = tf.layers.batch_normalization(distributed_deconv2d_1)

            # DECONV2
            # filter/stride: (3,3)/(2,2)
            # (batch_size,1,32,32,128) -> (batch_size,1,64,64,32)
            distributed_deconv2d_2 = self._distributedDeConv2D(distributed_deconv2d_1,
                                                               output_channels=32,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   64, 64, 32],
                                                               k_h=3, k_w=3, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_2')
            distributed_deconv2d_2 = self._lrelu(distributed_deconv2d_2)
            distributed_deconv2d_2 = tf.layers.batch_normalization(distributed_deconv2d_2)

            # DECONV3
            # filter/stride: (5,5)/(2,2)
            # (batch_size,1,64,64,32) -> (batch_size,1,128,128,16)
            distributed_deconv2d_3 = self._distributedDeConv2D(distributed_deconv2d_2,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   128, 128, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_3')
            distributed_deconv2d_3 = self._lrelu(distributed_deconv2d_3)
            distributed_deconv2d_3 = tf.layers.batch_normalization(distributed_deconv2d_3)

            # DECONV4
            # filter/stride: (5,2)/(5,2)
            # (batch_size,1,128,128,16) -> (batch_size,1,256,256,16)
            distributed_deconv2d_4 = self._distributedDeConv2D(distributed_deconv2d_3,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   256, 256, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_4')
            distributed_deconv2d_4 = self._lrelu(distributed_deconv2d_4)
            distributed_deconv2d_4 = tf.layers.batch_normalization(distributed_deconv2d_4)

            # DECONV5
            # filter/stride: (3,3)/(1,1)
            # (batch_size,1,256,256,16) -> (batch_size,1,256,256,input_c)
            distributed_deconv2d_5 = self._distributedDeConv2D(distributed_deconv2d_4,
                                                               output_channels=self._c,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   256, 256, self._c],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='distributed_deconv2d_5')
            # distributed_deconv2d_5 = self._lrelu(distributed_deconv2d_5)
            # distributed_deconv2d_5 = tf.layers.batch_normalization(distributed_deconv2d_5)
            # [self._batch_size, time_length, 256, 256, input_c]
            self._reconstr_out = distributed_deconv2d_5


    def _old_build(self):
        # with tf.variable_scope("generator"):
        # TimeDistributed 就是将 Conv2D 分别运用到t帧上
        # CONV1
        distributed_conv2d_1 = self._distributedConv2D(self._x, output_channels=16, k_h=5,
                                                       k_w=5, s_h=2, s_w=2,
                                                       name='distributed_conv2d_1')
        # distributed_conv2d_1 = self._lrelu(distributed_conv2d_1)
        distributed_conv2d_1 = tf.layers.batch_normalization(distributed_conv2d_1)
        distributed_conv2d_1 = tf.nn.tanh(distributed_conv2d_1)

        # CONV2
        distributed_conv2d_2 = self._distributedConv2D(distributed_conv2d_1, output_channels=32,
                                                       k_h=5,
                                                       k_w=5, s_h=2, s_w=2,
                                                       name='distributed_conv2d_2')
        distributed_conv2d_2 = tf.layers.batch_normalization(distributed_conv2d_2)
        distributed_conv2d_2 = tf.nn.tanh(distributed_conv2d_2)


        # CONV3
        distributed_conv2d_3 = self._distributedConv2D(distributed_conv2d_2, output_channels=64,
                                                       k_h=3,
                                                       k_w=3, s_h=2, s_w=2,
                                                       name='distributed_conv2d_3')
        distributed_conv2d_3 = tf.layers.batch_normalization(distributed_conv2d_3)
        distributed_conv2d_3 = tf.nn.tanh(distributed_conv2d_3)


        # CONV4
        distributed_conv2d_4 = self._distributedConv2D(distributed_conv2d_3, output_channels=128,
                                                       k_h=3,
                                                       k_w=3, s_h=1, s_w=1,
                                                       name='distributed_conv2d_4')
        distributed_conv2d_4 = tf.layers.batch_normalization(distributed_conv2d_4)
        distributed_conv2d_4 = tf.nn.tanh(distributed_conv2d_4)


        # CONVLSTM
        # # temporal encoder-decoder
        convlstm1 = self._convlstm(distributed_conv2d_4, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_1")
        convlstm2 = self._convlstm(convlstm1, output_channels=64, kernel_size=[3, 3],
                                   name="conv_lstm_cell_2")
        # convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
        #                            name="conv_lstm_cell_3")
        # convlstm3_g = convlstm3[:, -1:-2:-1, ...]
        # # DECONV1
        # distributed_deconv2d_1 = self._distributedDeConv2D(convlstm3_g, output_channels=64,
        #                                                    flatten_output_shape=[
        #                                                        self._batch_size * self._time_length,
        #                                                        32, 32, 64],
        #                                                    k_h=3, k_w=3, s_h=1, s_w=1,
        #                                                    name='distributed_deconv2d_1')
        # distributed_deconv2d_1 = tf.layers.batch_normalization(distributed_deconv2d_1)
        # distributed_deconv2d_1 = tf.nn.tanh(distributed_deconv2d_1)

        """
        predict stream
        """
        with tf.variable_scope('predict'):
            convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_3")
            convlstm3_g = convlstm3[:, -1:-2:-1, ...]
            # DECONV1
            distributed_deconv2d_1 = self._distributedDeConv2D(convlstm3_g, output_channels=64,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   32, 32, 64],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='distributed_deconv2d_1')
            distributed_deconv2d_1 = tf.layers.batch_normalization(distributed_deconv2d_1)
            distributed_deconv2d_1 = tf.nn.tanh(distributed_deconv2d_1)
            # distributed_deconv2d_1 = tf.nn.dropout(distributed_deconv2d_1, 0.5)

            # DECONV2
            distributed_deconv2d_2 = self._distributedDeConv2D(distributed_deconv2d_1,
                                                               output_channels=32,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   64, 64, 32],
                                                               k_h=3, k_w=3, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_2')
            distributed_deconv2d_2 = tf.layers.batch_normalization(distributed_deconv2d_2)
            distributed_deconv2d_2 = tf.nn.tanh(distributed_deconv2d_2)
            # distributed_deconv2d_2 = tf.nn.dropout(distributed_deconv2d_2, 0.5)

            # DECONV3
            distributed_deconv2d_3 = self._distributedDeConv2D(distributed_deconv2d_2,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size,
                                                                   128, 128, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_3')
            distributed_deconv2d_3 = tf.layers.batch_normalization(distributed_deconv2d_3)
            distributed_deconv2d_3 = tf.nn.tanh(distributed_deconv2d_3)
            # distributed_deconv2d_3 = tf.nn.dropout(distributed_deconv2d_3, 0.5)

            # DECONV4
            distributed_deconv2d_4 = self._distributedDeConv2D(distributed_deconv2d_3,
                                                               output_channels=self._c,
                                                               flatten_output_shape=
                                                               [self._batch_size,
                                                                   256, 256, self._c],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_4')
            # distributed_deconv2d_4 = tf.layers.batch_normalization(distributed_deconv2d_4)
            distributed_deconv2d_4 = tf.nn.tanh(distributed_deconv2d_4)
            


            # [self._batch_size, 256, 256, 1]
            self._pred_out = tf.squeeze(distributed_deconv2d_4, axis=[1])

        """
        reconstruct stream
        """
        with tf.variable_scope('reconstruct'):
            convlstm3 = self._convlstm(convlstm2, output_channels=128, kernel_size=[3, 3],
                                   name="conv_lstm_cell_3")
            # DECONV1
            distributed_deconv2d_1 = self._distributedDeConv2D(convlstm3, output_channels=64,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   32, 32, 64],
                                                               k_h=3, k_w=3, s_h=1, s_w=1,
                                                               name='distributed_deconv2d_1')
            distributed_deconv2d_1 = tf.layers.batch_normalization(distributed_deconv2d_1)
            distributed_deconv2d_1 = tf.nn.tanh(distributed_deconv2d_1)

            # DECONV2
            distributed_deconv2d_2 = self._distributedDeConv2D(distributed_deconv2d_1,
                                                               output_channels=32,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   64, 64, 32],
                                                               k_h=3, k_w=3, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_2')
            distributed_deconv2d_2 = tf.layers.batch_normalization(distributed_deconv2d_2)
            distributed_deconv2d_2 = tf.nn.tanh(distributed_deconv2d_2)

            # DECONV3
            distributed_deconv2d_3 = self._distributedDeConv2D(distributed_deconv2d_2,
                                                               output_channels=16,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   128, 128, 16],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_3')
            distributed_deconv2d_3 = tf.layers.batch_normalization(distributed_deconv2d_3)
            distributed_deconv2d_3 = tf.nn.tanh(distributed_deconv2d_3)

            # DECONV4
            distributed_deconv2d_4 = self._distributedDeConv2D(distributed_deconv2d_3,
                                                               output_channels=self._c,
                                                               flatten_output_shape=[
                                                                   self._batch_size * self._time_length,
                                                                   256, 256, self._c],
                                                               k_h=5, k_w=5, s_h=2, s_w=2,
                                                               name='distributed_deconv2d_4')
            distributed_deconv2d_4 = tf.layers.batch_normalization(distributed_deconv2d_4)
            distributed_deconv2d_4 = tf.nn.tanh(distributed_deconv2d_4)

            # [self._batch_size, time_length, 256, 256, 1]
            self._reconstr_out = distributed_deconv2d_4


    def _convlstm(self, input, output_channels, kernel_size=[3, 3], name='convlstm'):
        """
        input: size is [batch_size, time_steps, h, w, c]
        cell input_shape: Shape of the input, excluding the batch size and time length,
                    [h,w,c]
        outputs: time_steps步里所有的输出。它的形状为(batch_size, time_steps, h, w, c)
        state是最后一步的隐状态，它的形状为(batch_size, h, w, c)
        """
        with tf.variable_scope(name):
            convlstm_cell = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[input.get_shape().as_list()[2], input.get_shape().as_list()[3],
                             input.get_shape().as_list()[4]],
                output_channels=output_channels,
                kernel_shape=kernel_size,
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name=name)

            initial_state = convlstm_cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)
            # 调用n次cell
            outputs, _ = tf.nn.dynamic_rnn(convlstm_cell, input, initial_state=initial_state,
                                           time_major=False, dtype="float32")
            return outputs


    def _distributedConv2D(self, input, output_channels, k_h=3, k_w=3, s_h=1, s_w=1, name='conv_h'):
        """
        input: size is [batch_size, time_steps, h, w, c]
        single_input_shape: Shape of the input as int tuple, excluding the batch size,
            [time_steps, h,w,c]
        outputs: time_steps步里所有的输出。它的形状为(batch_size, time_steps, h, w, c)
        state是最后一步的隐状态，它的形状为(batch_size, h, w, c)
        """
        time_length = input.get_shape().as_list()[1]
        # reshape input to [batch_size*time_steps, h, w, c]
        input = tf.reshape(input, [-1, input.get_shape().as_list()[2],
                                   input.get_shape().as_list()[3],
                                   input.get_shape().as_list()[4]])

        with tf.variable_scope(name):
            # The Glorot normal initializer, also called Xavier normal initializer.
            # filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
            k = tf.get_variable('conv2d', [k_h, k_w, input.get_shape().as_list()[-1],
                                           output_channels],
                                initializer=tf.glorot_normal_initializer())

            # Computes a 2-D convolution given 4-D input and filter tensors
            # input tensor of shape [batch, in_height, in_width, in_channels]
            # filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
            # output [batch_size, h, w, c]
            conv = tf.nn.conv2d(input, filter=k, strides=[1, s_h, s_w, 1], padding="SAME")
            b = tf.get_variable('biases', [1, 1, 1, output_channels],
                                initializer=tf.constant_initializer(0.0))
            output = conv+b
            output = tf.reshape(output, [-1, time_length, output.get_shape().as_list()[1],
                                         output.get_shape().as_list()[2],
                                         output.get_shape().as_list()[3]])
            return output


    def _distributedDeConv2D(self, input, output_channels, flatten_output_shape, k_h=3, k_w=3,
                             s_h=1, s_w=1, name='deconv_h'):
        """
        input: size is [batch_size, time_steps, h, w, c]
        single_input_shape: Shape of the input as int tuple, excluding the batch size,
            [time_steps, h,w,c]
        outputs: time_steps步里所有的输出。它的形状为(batch_size, time_steps, h, w, c)
        state是最后一步的隐状态，它的形状为(batch_size, h, w, c)
        """
        time_length = input.get_shape().as_list()[1]
        # reshape input to [batch_size*time_steps, h, w, c]
        input = tf.reshape(input, [-1, input.get_shape().as_list()[2],
                                   input.get_shape().as_list()[3],
                                   input.get_shape().as_list()[4]])

        with tf.variable_scope(name):
            # The Glorot normal initializer, also called Xavier normal initializer.
            # filter tensor of shape [filter_height, filter_width, out_channels, in_channels]
            k = tf.get_variable('deconv2d', [k_h, k_w, output_channels,
                                             input.get_shape().as_list()[-1]],
                                initializer=tf.glorot_normal_initializer())

            # Computes a 2-D convolution given 4-D input and filter tensors
            # input tensor of shape [batch, in_height, in_width, in_channels]
            # output [batch_size, h, w, c]
            deconv = tf.nn.conv2d_transpose(input, filter=k, output_shape=flatten_output_shape,
                                          strides=[1, s_h, s_w, 1],
                                          padding="SAME")
            b = tf.get_variable('biases', [1, 1, 1, output_channels],
                                initializer=tf.constant_initializer(0.0))
            output = deconv+b
            output = tf.reshape(output, [-1, time_length, output.get_shape().as_list()[1],
                                         output.get_shape().as_list()[2],
                                         output.get_shape().as_list()[3]])
            return output


    def _conv3D(self, input, output_channels, k_h=3, k_w=3, s_h=1, s_w=1, name='conv3d'):
        with tf.variable_scope(name):
            # The Glorot normal initializer, also called Xavier normal initializer.
            # filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
            k = tf.get_variable('conv2d', [k_h, k_w, input.get_shape().as_list()[-1],
                                           output_channels],
                                initializer=tf.glorot_normal_initializer())

            # Computes a 2-D convolution given 4-D input and filter tensors
            # input tensor of shape [batch, in_height, in_width, in_channels]
            # filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
            # output [batch_size, h, w, c]
            conv = tf.nn.conv2d(input, filter=k, strides=[1, s_h, s_w, 1], padding="SAME")
            b = tf.get_variable('biases', [1, 1, 1, output_channels],
                                initializer=tf.constant_initializer(0.0))
            output = conv+b
            output = tf.reshape(output, [-1, time_length, output.get_shape().as_list()[1],
                                         output.get_shape().as_list()[2],
                                         output.get_shape().as_list()[3]])
            return output


    def _lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)
