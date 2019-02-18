import tensorflow as tf
from tensorflow.keras import backend as K_B
import numpy as np
import argparse
import datetime
import logging
import time
import os
from models import CSRNet
from config_reader import Config
from dataset_util import data_iterator
from loss_functions import compute_euclidean_distance, compute_mse_error, compute_mae_error
from train_util import init_or_restore, average_gradients, average_losses, colorize
"""HYPER-PARAMETER"""
use_trick = 0
for_remote = 0
for_pc = 1
file_name = 'myCSRNet'

img_rows = 512
img_cols = 512
fac = 8
fine_tuned = 1

if for_remote == 1:
    """for remote"""
    """file_name need to be modified"""
    # dataset_root_dir = '/data/zyl/graduate'
    # exp_data_root_dir = '/data/zyl/graduate/exp_data/' + file_name
    # npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
    # tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
    # gt_root_dir = dataset_root_dir + '/cgan_data'
    # checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
    # summary_dir = exp_data_root_dir + '/summary'
    # regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
    # log_path = exp_data_root_dir + '/log'
    # gpu_nums = 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    pass
else:
    if for_pc == 1:
        """for local"""
        """file_name need to be modified"""
        dataset_root_dir = '/home/orli/Blue-HDD/1_final_lab_/Dataset'
        exp_data_root_dir = '/home/orli/Blue-HDD/1_final_lab_/exp_data/crowd_counting/' + file_name
        # npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
        tfrecord_root_dir = dataset_root_dir + '/crowd_conuting/tfrecord_dataset/ShanghaiTech'
        # gt_root_dir = dataset_root_dir + '/cgan_data'
        checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
        summary_dir = exp_data_root_dir + '/summary'
        # regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
        log_path = exp_data_root_dir + '/log'
        gpu_nums = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    else:
        """for notebook"""
        # dataset_root_dir = 'D:/study/337_lab/thesis/data'
        # exp_data_root_dir = 'D:/study/337_lab/thesis/exp_data/' + file_name
        # npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
        # tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
        # gt_root_dir = dataset_root_dir + '/cgan_data'
        # checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
        # summary_dir = exp_data_root_dir + '/summary'
        # regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
        # log_path = exp_data_root_dir + '/log'
        # FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
        # gpu_nums = 1
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        pass


# 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"


parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--dataset', dest='dataset', default='part_A', help='dataset name')
args = parser.parse_args()

"""
Train phase
"""
def train(cfg, logger, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_dir = os.path.join(checkpoint_dir, model_name)
    print('[!!!] model name:{}'.format(model_dir))
    logger.info('[!!!] model name:{}'.format(model_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, model_name)
    best_model_ckpt_dir = os.path.join(model_dir, 'best_model')
    
    if not os.path.exists(best_model_ckpt_dir):
        os.makedirs(best_model_ckpt_dir)
    best_ckpt_path = os.path.join(best_model_ckpt_dir, 'best_model')
    
    trick_model_ckpt_dir = os.path.join(model_dir, 'trick_model')
    if not os.path.exists(trick_model_ckpt_dir):
        os.makedirs(trick_model_ckpt_dir)
    trick_ckpt_path = os.path.join(trick_model_ckpt_dir, 'trick_model')

    lr = cfg.lr

    with tf.device('/cpu:0'):
        g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')

        assert cfg.opt in ['Adam', 'SGD'], '[!!!] wrong optimizer name'
        if cfg.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')
        elif cfg.opt == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, name='optimizer')
        else:
            print('[!!!] wrong optimizer name')

        small_batch = cfg.batch_size // gpu_nums
        iterator = data_iterator(args.dataset, 'train', cfg, small_batch,
                                 tfrecord_root_dir=tfrecord_root_dir, logger=logger)
        # generator_tower_grads
        tower_grads = []

        mae_losses = []
        mse_losses = []
        losses = []

        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    images, labels = iterator.get_next()
                    # labels_resized = tf.image.resize_images(labels,
                    #                                         [img_rows // fac, img_cols // fac])

                    with tf.variable_scope('model', reuse=(i>0)):
                        model_b = CSRNet(cfg, images, small_batch, 'b')
                        # [batch, h, w, c]

                    outputs = model_b.output
                    if i == 0:
                        # print model
                        model_b.full_model.summary()
                        model_b.full_model.summary(print_fn=logger.info)

                    loss = compute_euclidean_distance(outputs, labels)
                    losses.append(loss)
                    mae_loss = compute_mae_error(outputs, labels)
                    mae_losses.append(mae_loss)
                    mse_loss = compute_mse_error(outputs, labels)
                    mse_losses.append(mse_loss)

                    # # 重用variable
                    tf.get_variable_scope().reuse_variables()

                    # add summaries
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    summaries.append(tf.summary.scalar(tensor=loss,
                                                       name='loss'))
                    summaries.append(tf.summary.scalar(tensor=mae_loss,
                                                       name='mae_loss'))
                    summaries.append(tf.summary.scalar(tensor=mse_loss,
                                                       name='mse_loss'))
                    summaries.append(tf.summary.image(tensor=images, name='images'))
                    summaries.append(tf.summary.image(tensor=tf.map_fn(lambda img:
                                                                       colorize(img, cmap='jet'),
                                                                       labels),
                                                      name='label'))
                    summaries.append(tf.summary.image(tensor=tf.map_fn(lambda img:
                                                                       colorize(img, cmap='jet'),
                                                                       tf.image.resize_images(
                                                                           outputs,
                                                                           [224, 224])),
                                                      name='outputs'))

                    if fine_tuned == 1:
                        vars = [var for var in tf.trainable_variables() if "dil" in var.name]
                    else:
                        train_vars = tf.trainable_variables()
                        vars = [var for var in train_vars if "model" in var.name]

                        # equal: vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                        #                           scope='tower_{}/model'.format(i))

                    grads = optimizer.compute_gradients(loss, var_list=vars)
                    tower_grads.append(grads)

        # 计算所有loss
        average_loss = average_losses(losses)
        average_mae_loss = average_losses(mae_losses)
        average_mse_loss = average_losses(mse_losses)
        # cpu 上计算平均梯度
        grads = average_gradients(tower_grads)

        # 更新
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # calculate gradients
            train_op = optimizer.apply_gradients(grads, global_step=g_step)

        # add history for variables and gradients in genrator
        for var in vars:
            summaries.append(
                tf.summary.histogram('Model/' + var.op.name, var)
            )

        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram('Model/' + var.op.name + '/gradients', grad)
                )

        # create a saver
        saver = tf.train.Saver(max_to_keep=1)
        # saver_for_best = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        saver_for_best = tf.train.Saver(max_to_keep=3)
        if use_trick:
            saver_for_trick = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        # build summary
        summary_op = tf.summary.merge(summaries)

    # start training session
    # "allow_soft_placement" must be set to True to build towers on GPU,
    # as some of the ops do not have GPU implementations.
    # "log_device_placement" set to True will print device place
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    if for_remote != 1:
        # fraction of overall amount of memory that each GPU should be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

    tf_sess = tf.Session(config=config)
    K_B.set_session(tf_sess)
    with K_B.get_session() as sess:
        # summaries
        summary_path = os.path.join(summary_dir, model_name)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        init_or_restore(sess, saver, ckpt_path, logger)

        volumes_per_step = small_batch * gpu_nums

        step = -1
        min_train_loss = np.inf
        min_train_loss_step = 0
        patient = 0
        if type(cfg.epochs) is int:
            for epoch in range(cfg.epochs):
                print("EPOCH: {}".format(epoch + 1))
                logger.info("EPOCH: {}".format(epoch + 1))
                sess.run(iterator.initializer)
                while True:
                    try:
                        step += 1
                        # tf.cod的validate分支也会get_next， 避免先用完导致跳出循环
                        _, average_loss_v, average_mae_loss_v, average_mse_loss_v, g_step_v, \
                        summary_str = sess.run(
                            [train_op, average_loss, average_mae_loss, average_mse_loss, g_step,
                             summary_op])

                        assert (not np.isnan(average_loss_v)), 'Model diverged with ' \
                                                                 'loss = NaN'

                        # duration = time.time() - start_time
                        # batch_per_sec = volumes_per_step / duration

                        if step % 10 == 0:
                            print("----- step:{} train loss:{:04f}".format(
                                step, average_loss_v))
                            print("----- step:{} train mae_loss:{:04f}".format(
                                step, average_mae_loss_v))
                            print("----- step:{} train mse_loss:{:04f}".format(
                                step, average_mse_loss_v))
                            logger.info("----- step:{} train loss:{:04f}".format(
                                step, average_loss_v))
                            logger.info("----- step:{} train mae_loss:{:04f}".format(
                                step, average_mae_loss_v))
                            logger.info("----- step:{} train mse_loss:{:04f}".format(
                                step, average_mse_loss_v))
                        if step % 100 == 0:
                            summary_writer.add_summary(summary_str, step)
                        if step % 1000 == 0:
                            saver.save(sess, ckpt_path, global_step=step)
                    except tf.errors.OutOfRangeError:
                        print('train dataset finished')
                        logger.info('train dataset finished')
                        break
                    # catch all other error
                    except BaseException as e:
                        print('[!!!]An exception occurred: {}'.format(e))

                if average_loss_v < min_train_loss:
                    min_train_loss = average_loss_v
                    min_train_loss_step = step
                    patient = 0
                    saver_for_best.save(sess, best_ckpt_path, global_step=step)
                    print("update best model, min_train_loss is [{:04f}] in step [{}]".format(
                        average_loss_v, step))
                    logger.info("update best model, min_train_loss is [{:04f}] in step [{}]".format(
                        average_loss_v, step))
                else:
                    patient += 1
                    if patient >= cfg.patient:
                        print('[!!!] Early stop for no reducing in train loss ' +
                              'after [{}] epochs, '.format(cfg.patient) +
                              'min_train_loss is [{:04f}]'.format(min_train_loss) +
                              ', in step [{}]'.format(min_train_loss_step))
                        print('mae_loss is [{:04f}]'.format(average_mae_loss_v))
                        print('mse_loss is [{:04f}]'.format(average_mse_loss_v))
                        logger.warning('[!!!] Early stop for no reducing in train loss ' +
                                       'after [{}] epochs, '.format(cfg.patient) +
                                       'min_train_loss is [{:04f}]'.format(min_train_loss) +
                                       ', in step [{}]'.format(min_train_loss_step))
                        logger.warning('mae_loss is [{:04f}]'.format(average_mae_loss_v))
                        logger.warning('mse_loss is [{:04f}]'.format(average_mse_loss_v))
                    break

    print('[!!!] model name:{}'.format(model_name))
    logger.info('[!!!] model name:{}'.format(model_name))


"""
Test phase
"""
def test(cfg, logger, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_dir = os.path.join(checkpoint_dir, model_name)
    best_model_ckpt_dir = os.path.join(model_dir, 'best_model')
    assert os.path.exists(best_model_ckpt_dir)

    batch_size = tf.placeholder(tf.int64, [], name='batch_size')
    iterator = data_iterator(args.dataset, 'test', cfg, batch_size,
                             tfrecord_root_dir=tfrecord_root_dir, logger=logger)

    images, labels = iterator.get_next()

    with tf.variable_scope('model'):
        model_b = CSRNet(cfg, images, batch_size, 'b')

    print(model_b.model_summary)
    # [batch, h, w, c]
    outputs = model_b.output

    # comput loss
    loss = compute_euclidean_distance(outputs, labels)
    mae_loss = compute_mae_error(outputs, labels)
    mse_loss = compute_mse_error(outputs, labels)

    # saver_for_best = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
    saver_for_best = tf.train.Saver(max_to_keep=3)


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    if for_remote != 1:
        # fraction of overall amount of memory that each GPU should be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with tf.Session(config=config) as sess:
        init_or_restore(sess, saver_for_best, best_model_ckpt_dir, logger)

        start_time = time.time()
        sess.run(iterator.initializer)

        losses = []
        mae_losses = []
        mse_losses = []
        while True:
            try:
                loss_v, mae_loss_v, mse_loss_v= sess.run(
                            [loss, mae_loss, mse_loss],
                            feed_dict={batch_size:1})
                losses.append(loss_v)
                mae_losses.append(mae_loss_v)
                mse_losses.append(mse_loss_v)
            except tf.errors.OutOfRangeError:
                print('test dataset finished')
                logger.info('test dataset finished')
                break
            # catch all other error
            except BaseException as e:
                print('[!!!]An exception occurred: {}'.format(e))

        average_loss = average_losses(losses)
        average_mae_loss = average_losses(mae_losses)
        average_mse_loss = average_losses(mse_losses)

        print('mae_loss is [{:04f}]'.format(average_loss))
        print('mae_loss is [{:04f}]'.format(average_mae_loss))
        print('mse_loss is [{:04f}]'.format(average_mse_loss))
        print('[!!!] model name:{}'.format(model_name))
        logger.info('[!!!] model name:{}'.format(model_name))

"""
Test_in_train phase
"""
def test_in_train(cfg, logger, sess, model_name):
    pass


def main(*argc, **argv):
    cfg = Config()

    model_name = '{}_{}-{}-{}_{}-{}-{}'.format(args.dataset, cfg.augment,
                                               cfg.batch_size, cfg.opt, cfg.lr,
                                         cfg.epochs, cfg.patient)
    if cfg.quick_train == 0:
        model_name += '-full'

    log_dir = os.path.join(log_path, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=os.path.join(log_dir, '{}.log'.format(datetime.datetime.now().
                                                                        strftime('%Y%m%d-%H%M%S'))),
                        level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger()

    if args.phase == 'train':
        train(cfg, logger, model_name)
        test(cfg, logger, model_name)
    elif args.phase == 'test':
        test(cfg, logger, model_name)
    else:
        print('[!!!] wrong key word')

        
if __name__  == "__main__":
    tf.app.run()
