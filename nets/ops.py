import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.slim as slim

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding, name="outs")

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x



def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.
  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.
  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.
  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.
  Note that
     net = conv2d_same(inputs, num_outputs, 3, stride=stride)
  is equivalent to
     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)
  whereas
     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.
  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(
        tensor=inputs,
        paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0') 
        # x = tf.layers.batch_normalization(x_init, training=is_training)   
        x = relu(x)

        if downsample :
            x = conv2d_same(x, channels, 3, stride=2, scope='conv_0')
            x_init = conv2d_same(x_init, channels, 1, stride=2, scope='conv_init')

        else :
            x = conv2d_same(x, channels, 3, stride=1, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        # x = tf.layers.batch_normalization(x, training=is_training)     
        x = relu(x)
        x = conv2d_same(x, channels, 3, stride=1, scope='conv_1')

        return x + x_init


def resblock_org(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0') 
        # x = tf.layers.batch_normalization(x_init, training=is_training)   
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        # x = tf.layers.batch_normalization(x, training=is_training)     
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init

def resblock_kernel(x_init, channels, kernels=[3, 3], is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0') 
        # x = tf.layers.batch_normalization(x_init, training=is_training)   
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=kernels[0], stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=kernels[0], stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        # x = tf.layers.batch_normalization(x, training=is_training)     
        x = relu(x)
        x = conv(x, channels, kernel=kernels[1], stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x, scope):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True, name=scope)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)
    
##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy



def residual_block(net, block, repeat, name, use_stride=True, is_training=None):
    # print("block_%s" % name)
    for i in range(repeat):
        short_cut = net
        for j, filter in enumerate(block):
            stride = 1
            if i == 0 and j == 0 and use_stride:
                stride = 2
            net = tf.layers.conv2d(net, filter[1], filter[0], stride, 'same', name="%s_%d_%d" % (name, i, j),
                                   use_bias=False)
            net = tf.layers.batch_normalization(net, training=is_training)
            # print(net)
            if j > len(block) - 1:
                net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
        short_cut_channel = short_cut.get_shape()[3]
        last_layer_channel = net.get_shape()[3]

        stride = 1
        if i == 0 and use_stride:
            stride = 2

        if short_cut_channel == last_layer_channel:
            if stride > 1:
                short_cut = tf.layers.max_pooling2d(short_cut, 1, strides=stride)
        else:
            short_cut = tf.layers.conv2d(short_cut, int(net.get_shape()[3]), 1, stride, 'same',
                                         name="%s_projection_%d_%d" % (name, i, j))
        net += short_cut
        net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
    return net

def conv_bn_layer(name, bottom, is_training, num_output,
                  kernel_size, stride, is_bn=True, is_act=True, m_padding='SAME'):
    bottom = tf.convert_to_tensor(bottom)
    num_input = bottom.get_shape()[-1].value
    with tf.name_scope(name):
        weights = tf.get_variable(name+'_w',
                                  shape=[kernel_size, kernel_size, num_input, num_output],
                                  dtype=tf.float32,
                                  initializer=tf.glorot_normal_initializer())
        conv = tf.nn.conv2d(bottom, weights, [1, stride, stride, 1], padding=m_padding)
        
        bias = tf.get_variable(name + '_b',
                               shape=[num_output],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0))

        conv_b = tf.nn.bias_add(conv, bias)

        # whether use Batch Normalization
        if is_bn is True:
            # conv_b = bn_layer(conv_b, is_training, name=name+'_bn')
            conv_b = batch_norm(conv_b, is_training, scope=name+'_bn')

        # whether use ReLU activation
        if is_act is True:
            conv_b = tf.nn.relu(conv_b, name=name+'_relu')

        return conv_b




