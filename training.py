# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
from tfdata import *
import numpy as np
import tensorflow as tf


# In[ ]:

def weight_variable(shape, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name='weights',
                              shape=shape,
                              trainable=True,
                              initializer=tf.truncated_normal_initializer(stddev=0.01))

        ### L2-regularization
        REGULARIZATION_RATE = 0.0001
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        tf.add_to_collection('losses', regularizer(weights))

        return weights


# In[ ]:


def bias_variable(shape,name):
    with tf.variable_scope(name) as scope:
        biases = tf.get_variable(name='biases',
                                 shape=shape,
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.01))

        return biases

def conv2d(input, in_feature_dim, out_feature_dim, kernel_size, with_bias=False, name=None):
    W = weight_variable([kernel_size, kernel_size, in_feature_dim, out_feature_dim], name=name)
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_feature_dim], name=name)
    return conv


# In[ ]:


def batch_activ_conv(current, in_feature_dim, out_feature_dim, kernel_size, is_training, keep_prob, name):
    with tf.variable_scope(name) as scope:
        # current = tf.contrib.layers.batch_norm(current, decay=0.9, scale=True, is_training=False,
        #                                       updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name)
        if is_training:
            current = tf.contrib.layers.batch_norm(current, decay=0.9, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name)
        else:
            current = tf.contrib.layers.batch_norm(current, decay=0.9, scale=True, is_training=is_training, scope=name)
        current = tf.nn.relu(current)
        current = conv2d(current, in_feature_dim, out_feature_dim, kernel_size, name=name)
        current = tf.nn.dropout(current, keep_prob)
        return current


# In[ ]:


### growth: feature maps that each layer preduce, equals to the number of filters
def block(input, layers, in_feature_dim, growth, is_training, keep_prob, name):
    current = input
    sum_feature_dim = in_feature_dim
    for id in range(layers):
        tmp = batch_activ_conv(current, sum_feature_dim, growth, 3, is_training, keep_prob,
                               name=name + '/layer_%d' % id)
        current = tf.concat((current, tmp), 3)
        sum_feature_dim += growth
    return current, sum_feature_dim


# In[ ]:


def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')


# In[ ]:


def loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    # l2-regularization
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy') + tf.add_n(tf.get_collection('losses'))

    return cross_entropy_mean


# In[ ]:


# Train step
def train(loss_value, model_learning_rate):
    # Create optimizer
    # my_optimizer = tf.train.MomentumOptimizer(model_learning_rate, momentum=0.9)

    my_optimizer = tf.train.AdamOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return train_step


# In[ ]:


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


# In[ ]:


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding="bytes").item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    get_var = tf.get_variable(subkey).assign(data)
                    session.run(get_var)


# In[ ]:


def fc(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        Wfc = weight_variable([num_in, num_out], name=name)
        bfc = bias_variable([num_out], name=name)

        tf.summary.histogram(name + "/weights", Wfc)
        tf.summary.histogram(name + "/biases", bfc)

        act = tf.nn.xw_plus_b(x, Wfc, bfc, name=name + '/op')

        return act


# In[ ]:


def DenseNet(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 32, 32, 3])
    current = conv2d(current, 3, 72, 3, name='preprocessing')
    ###  Dense40 original  the second and fourth parameters of each block are 12,12, respectively
    ### denseblock1
    current, featurenumber = block(current, 12, 72, 36, is_training, keep_prob, name='denseblock1')
    ### transition layer1
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2)

    ### denseblock2
    current, featurenumber = block(current, 12, featurenumber, 36, is_training, keep_prob, name='denseblock2')
    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2)

    ### denseblock3
    current, featurenumber = block(current, 12, featurenumber, 36, is_training, keep_prob, name='denseblock3')

    current = tf.contrib.layers.batch_norm(current, decay=0.9, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 8)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 21, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output


#### loss 直接输入current和label即可


# In[ ]:


def main():
    # Dataset path
    train_tfrecords = 'train.tfrecords'
    test_tfrecords = 'test.tfrecords'

    # Learning params  原来imagenet的学习率是0.001
    learning_rate = 0.0001
    training_iters = 33600 # 一个epoch两千次
    batch_size = 40

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size)

    # Model
    with tf.variable_scope('model_definition') as scope:
        train_output = DenseNet(train_img, is_training=True, keep_prob=0.5)
        scope.reuse_variables()
        test_output = DenseNet(test_img, is_training=False, keep_prob=1)

    # Loss and optimizer
    loss_op = loss(train_output, train_label)
    # this aims to test whether or not the model is overfitting  to check loss value of test samples
    test_loss_op=loss(test_output,test_label)

    tf.summary.scalar('loss', loss_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = train(loss_op, learning_rate)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_output, train_label)
    tf.summary.scalar("train_accuracy", train_accuracy)

    test_accuracy = accuracy_of_batch(test_output, test_label)
    tf.summary.scalar("test_accuracy", test_accuracy)

    # Init
    init = tf.global_variables_initializer()

    # Summary
    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    # saver = tf.train.Saver(tf.trainable_variables())

    ### new solution
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list)



    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)
        # with tf.variable_scope('model_definition'):
        #      load_with_skip('bvlc_alexnet.npy', sess, ['fc'])

        # load_ckpt_path = 'checkpoint/my-model.ckpt-33600'
        # saver.restore(sess, load_ckpt_path)

        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        print('Start training')
        # coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        for step in range(training_iters):
            step += 1
            #   _, loss_value = sess.run([train_op, loss_op])
            #   print('Generation {}: Loss = {:.5f}'.format(step, loss_value))

            ### this aims to test whether or not the model is overfitting  to check loss value of test samples
            _, loss_value, test_loss_value = sess.run([train_op, loss_op,test_loss_op])
            print('Generation {}: Loss = {:.5f}     Test Loss={:.5f}'.format(step, loss_value, test_loss_value))

            # print(Wfc1value[1, 1], Wfc2value[1, 1])

            # Display testing status
            if step % 40 == 0:
                acc1 = sess.run(train_accuracy)
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                acc2 = sess.run(test_accuracy)
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))

            if step % 40 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 840 == 0:
                saver.save(sess, 'checkpoint/my-model.ckpt', global_step=step)

        print("Finish Training and validation!")

        # coord.request_stop()
        # coord.join(threads)


# In[ ]:


if __name__ == '__main__':
    main()

