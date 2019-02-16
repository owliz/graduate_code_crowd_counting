import tensorflow as tf
import numpy as np
import argparse
import datetime
import logging
import time
import os
from models import Model
from config_reader import Config
from dataset_util import data_iterator
from loss_functions import compute_mse_error
from train_util import init_or_restore, average_gradients, average_losses, colorize
"""HYPER-PARAMETER"""
use_trick = 0
for_remote = 1
for_pc = 1
file_name = 'myCSRNet'

img_rows = 512
img_cols = 512
fac = 8

if for_remote == 1:
    """for remote"""
    """file_name need to be modified"""
    dataset_root_dir = '/data/zyl/graduate'
    exp_data_root_dir = '/data/zyl/graduate/exp_data/' + file_name
    npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
    tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
    gt_root_dir = dataset_root_dir + '/cgan_data'
    checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
    summary_dir = exp_data_root_dir + '/summary'
    regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
    log_path = exp_data_root_dir + '/log'
    FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
    gpu_nums = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
else:
    if for_pc == 1:
        """for local"""
        """file_name need to be modified"""
        dataset_root_dir = '/home/orli/Blue-HDD/1_final_lab_/Dataset'
        exp_data_root_dir = '/home/orli/Blue-HDD/1_final_lab_/exp_data/' + file_name
        npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
        tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
        gt_root_dir = dataset_root_dir + '/cgan_data'
        checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
        summary_dir = exp_data_root_dir + '/summary'
        regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
        log_path = exp_data_root_dir + '/log'
        FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
        gpu_nums = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    else:
        """for notebook"""
        dataset_root_dir = 'D:/study/337_lab/thesis/data'
        exp_data_root_dir = 'D:/study/337_lab/thesis/exp_data/' + file_name
        npy_root_dir = dataset_root_dir + '/cgan_data/npy_dataset'
        tfrecord_root_dir = dataset_root_dir + '/cgan_data/tfrecord_dataset'
        gt_root_dir = dataset_root_dir + '/cgan_data'
        checkpoint_dir = exp_data_root_dir + '/training_checkpoints'
        summary_dir = exp_data_root_dir + '/summary'
        regularity_score_root_dir = exp_data_root_dir + '/regularity_score'
        log_path = exp_data_root_dir + '/log'
        FLOWNET_CHECKPOINT = dataset_root_dir + '/cgan_data/pretrained_flownet/flownet-SD.ckpt-0'
        gpu_nums = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
    
    with tf.device('/cpu:0'):
        g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')

        if cfg.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=cfg.lr, name='optimizer')
        elif cfg.opt == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr, name='optimizer')
        else:
            print('[!!!] wrong optimizer name')
        small_batch = cfg.batch_size // gpu_nums

        iterator = data_iterator(args.dataset, cfg, small_batch,
                                 tfrecord_root_dir=tfrecord_root_dir, logger=logger)

        # generator_tower_grads
        tower_grads = []
        losses = []

        batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    images, labels = iterator.get_next()
                    labels_resized = tf.image.resize_images(labels,
                                                            [img_rows // fac, img_cols // fac])

                    with tf.variable_scope('generator', reuse=(i>0)):
                        model_b = Model(cfg, images, batch_size, 'b')
                        # [batch, h, w, c]
                        outputs = model_b._out

                    loss = compute_mse_error(outputs, labels)
                    losses.append(loss)

                    # # 重用variable
                    tf.get_variable_scope().reuse_variables()

                    # add summaries
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    summaries.append(tf.summary.scalar(tensor=loss,
                                                       name='loss'))
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
                        
                    vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope='model')
                    # g_vars = [x for x in g_vars if 'reconstruct' not in x.op.name]
                    generator_grads = optimizer.compute_gradients(loss, var_list=vars)
                    tower_grads.append(generator_grads)

        # 计算所有loss
        average_loss = average_losses(losses)
        # cpu 上计算平均梯度
        grads = average_gradients(tower_grads)

        # 更新
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # calculate gradients
            train_g_op = optimizer.apply_gradients(grads, global_step=g_step)

        # add history for variables and gradients in genrator
        for var in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='model'):
            summaries.append(
                tf.summary.histogram('Model/' + var.op.name, var)
            )
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram('Model/' + var.op.name + '/gradients', grad)
                )

        # create a saver
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        saver_for_best = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
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

    with tf.Session(config=config) as sess:
        # summaries
        summary_path = os.path.join(summary_dir, model_name)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        init_or_restore(sess, saver, ckpt_path, logger)

        if lam_flow != 0 and use_tvnet == 0:
            # initialize flownet
            initialize_flownet(sess, FLOWNET_CHECKPOINT)

        volumes_per_step = small_batch * gpu_nums

        step = -1
        min_val_g_loss = np.inf
        min_val_loss_step = 0
        if use_trick == 1:
            min_val_loss_auc = 0
            max_psnr_auc = -np.inf
            best_auc_step = 0
        patient = 0
        if type(cfg.epochs) is int:
            for epoch in range(cfg.epochs):
                print("EPOCH: {}".format(epoch + 1))
                logger.info("EPOCH: {}".format(epoch + 1))
                sess.run(train_iter.initializer)
                sess.run(val_iter.initializer)
                while True:
                    try:
                        step += 1
                        # tf.cod的validate分支也会get_next， 避免先用完导致跳出循环
                        sess.run(val_iter.initializer)
                        # 一个step会并行运行所有的gpu
                        # 注意这里run的是cpu上的总的操作
                        # start_time = time.time()
                        if adversarial:
                            # print('Training discriminator...')
                            # logger.info('Training discriminator...')
                            if GAN_type == 'LSGAN':
                                _, _, average_d_loss_v, d_step_v = sess.run(
                                    [clip_D, train_d_op, average_d_loss, d_step],
                                    feed_dict={is_training: True,
                                               batch_size: cfg.batch_size // gpu_nums})
                                assert (not np.isnan(average_d_loss_v)), 'Model diverged with ' \
                                                                         'discriminator loss = NaN'
                            else:
                                _, average_d_loss_v, d_step_v = sess.run(
                                    [train_d_op, average_d_loss, d_step],
                                    feed_dict={is_training: True,
                                               batch_size: cfg.batch_size // gpu_nums})
                                assert (not np.isnan(average_d_loss_v)), 'Model diverged with ' \
                                                                         'discriminator loss = NaN'

                        # print('Training generator...')
                        # logger.info('Training generator...')
                        _, average_g_loss_no_rec_v, average_g_loss_v, g_step_v, summary_str = sess.run(
                            [train_g_op, average_g_loss_no_rec, average_loss, g_step, summary_op],
                            feed_dict={is_training: True,
                                       batch_size: cfg.batch_size // gpu_nums})

                        assert (not np.isnan(average_g_loss_v)), 'Model diverged with ' \
                                                                 'generator loss = NaN'
                        # duration = time.time() - start_time
                        # batch_per_sec = volumes_per_step / duration

                        if step % 10 == 0:
                            print("----- step:{} generator loss:{:09f}".format(
                                step, average_g_loss_v))
                            print("----- step:{} generator loss_no_rec:{:09f}".format(
                                step, average_g_loss_no_rec_v))
                            print("----- step:{} discriminator loss:{:09f}".format(
                                step, average_d_loss_v))
                            logger.info("----- step:{} generator loss:{:09f}".format(
                                step, average_g_loss_v))
                            logger.info("----- step:{} generator loss_no_rec:{:09f}".format(
                                step, average_g_loss_no_rec_v))
                            logger.info("----- step:{} discriminator loss:{:09f}".format(
                                step, average_d_loss_v))
                        if step % 100 == 0:
                            summary_writer.add_summary(summary_str, step)
                        if step % 1000 == 0:
                            saver.save(sess, ckpt_path, global_step=step)
                    except tf.errors.OutOfRangeError:
                        print('train dataset finished')
                        logger.info('train dataset finished')
                        break
                    except:
                        import sys
                        print("Unexpected error:", sys.exc_info())
                # 这样写还是tf.cond的锅，两个batch都要有
                sess.run(train_iter.initializer)
                sess.run(val_iter.initializer)
                print('[!] start validate model ...')
                logger.info('[!] start validate model ...')
                val_g_losses = []
                val_g_losses_no_rec = []
                while True:
                    try:
                        average_g_loss_no_rec_v, average_g_loss_v = sess.run([average_g_loss_no_rec,
                                                                average_loss], 
                                                                feed_dict={is_training: False,
                                       batch_size: cfg.batch_size // gpu_nums})

                        assert (not np.isnan(average_g_loss_v)), 'Model diverged with ' \
                                                                 'generator loss = NaN'
                        val_g_losses.append(average_g_loss_v)
                        val_g_losses_no_rec.append(average_g_loss_no_rec_v)
                    except tf.errors.OutOfRangeError:
                        break
                
                if patient_use_g_loss_no_rec == 1:
                    for loss in val_g_losses_no_rec:
                        tmp = []
                    tmp.append(np.expand_dims(loss, axis=0))
                else:
                    for loss in val_g_losses:
                        tmp = []
                        tmp.append(np.expand_dims(loss, axis=0))

                stacked_loss = np.concatenate(tuple(tmp), axis=0)
                cur_val_g_loss = np.mean(stacked_loss, axis=0)
                assert (not np.isnan(cur_val_g_loss)), 'Model diverged with cur_val_loss = NaN'
                print("----- step:[{}] validate loss:{:09f}".format(step, cur_val_g_loss))
                logger.info("----- step:{} validate loss:{:09f}".format(step, cur_val_g_loss))
                if cur_val_g_loss < min_val_g_loss:
                    min_val_g_loss = cur_val_g_loss
                    min_val_loss_step = step
                    patient = 0
                    saver_for_best.save(sess, best_ckpt_path, global_step=step)
                    print("update best model, min_val_g_loss is [{:09f}] in step [{}]".format(
                        cur_val_g_loss, step))
                    logger.info("update best model, min_val_g_loss is [{:09f}] in step [{}]".format(
                        cur_val_g_loss, step))
                    if use_trick == 1:
                        new_psnr_auc = test_in_train(cfg, logger, sess, model_name)
                        if step == min_val_loss_step:
                            min_val_loss_auc = new_psnr_auc
                            print("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                    min_val_loss_auc, step))
                            logger.info("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                min_val_loss_auc, step))
                        if new_psnr_auc > max_psnr_auc:
                            max_psnr_auc = new_psnr_auc
                            best_auc_step = step
                            print("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, step))
                            logger.info("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, 
                            step))
                            saver_for_trick.save(sess, trick_ckpt_path, global_step=step)
                else:
                    patient += 1
                    if patient >= cfg.patient:
                        print('[!!!] Early stop for no reducing in validate loss ' +
                                       'after [{}] epochs, '.format(cfg.patient) +
                                       'min_val_g_loss is [{:09f}]'.format(min_val_g_loss) + 
                                       ', in step [{}]'.format(min_val_loss_step))
                        logger.warning('[!!!] Early stop for no reducing in validate loss ' +
                                       'after [{}] epochs, '.format(cfg.patient) +
                                       'min_val_g_loss is [{:09f}]'.format(min_val_g_loss) + 
                                       ', in step [{}]'.format(min_val_loss_step))
                        # saver.save(sess, best_ckpt_path, global_step=step)
                        if use_trick == 1:
                            print("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                    min_val_loss_auc, min_val_loss_step))
                            logger.info("min_val_loss's auc is [{:09f}] in step [{}]".format(
                                min_val_loss_auc, min_val_loss_step))
                            print("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, 
                                best_auc_step))
                            logger.info("best auc is [{:09f}] in step [{}]".format(max_psnr_auc, 
                                best_auc_step))
                        break
    print('[!!!] model name:{}'.format(model_name))
    logger.info('[!!!] model name:{}'.format(model_name))
    print('patient_use_g_loss_no_rec={}'.format(patient_use_g_loss_no_rec))
    logger.info('patient_use_g_loss_no_rec={}'.format(patient_use_g_loss_no_rec))


def compute_and_save_scores(errors, dir, video_id, error_name):
    # regularity score
    scores = errors - min(errors)
    scores = scores / max(scores)
    if error_name != 'psnr':
        # fei psnr为 yichang常值，转为正常值
        scores = 1 - scores
    regularity_score_file_dir = os.path.join(dir, error_name)
    if not os.path.exists(regularity_score_file_dir):
        os.makedirs(regularity_score_file_dir)
    regularity_score_file_path = os.path.join(regularity_score_file_dir,
                                        'scores_{:02d}.txt'.format(video_id+1))
    np.savetxt(regularity_score_file_path, scores)


def save_pixel_loss(errors, dir, video_id, error_name):

    regularity_score_file_dir = os.path.join(dir, error_name)
    if not os.path.exists(regularity_score_file_dir):
        os.makedirs(regularity_score_file_dir)
    np.save(os.path.join(regularity_score_file_dir, 'losses_{:02d}.npy'.format(video_id+1)),
            errors)
"""
Test phase
"""
def test(cfg, logger, model_name):
    size = '{}_{}'.format(cfg.width, cfg.height)
    if use_rgb == 1:
        npy_dir = os.path.join(npy_root_dir, args.dataset, size, 'rgb_testing_frames')
    else:
        npy_dir = os.path.join(npy_root_dir, args.dataset, size, 'testing_frames')
    video_nums = len(os.listdir(npy_dir))

    regularity_score_dir = os.path.join(regularity_score_root_dir, args.dataset, model_name, 
                                            'testing_frames')
    if not os.path.exists(regularity_score_dir):
        os.makedirs(regularity_score_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_dir = os.path.join(checkpoint_dir, model_name)
    best_model_ckpt_dir = os.path.join(model_dir, 'best_model')
    assert os.path.exists(best_model_ckpt_dir)

    # [batch, clip_length, h, w, c]
    if use_rgb == 1:
        data = tf.placeholder(tf.float32, shape=[None, cfg.clip_length, cfg.height, cfg.width, 3])
    else:
        data = tf.placeholder(tf.float32, shape=[None, cfg.clip_length, cfg.height, cfg.width, 1])
    # [batch, clip_length-1,h, w, c]
    input = data[:, :-1, ...]
    # 最后一帧， [batch, h, w, c]
    input_gt = tf.squeeze(data[:, -1:-2:-1, ...], axis=[1])
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')

    with tf.variable_scope('generator'):
        generator = Generator(cfg, input, batch_size, use_rgb, use_skip)
        # [batch, h, w, c]
        pred_output = generator._pred_out
        if lam_two_stream != 0:
            reconstr_output = generator._reconstr_out
                        

    saver_for_best = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)

    # comput loss
    psnr = compute_psnr(gen_frames=pred_output, gt_frames=input_gt)
    mse_error = compute_mse_error(gen_frames=pred_output, gt_frames=input_gt)

    # compute pixle loss
    pixel_loss = compute_pixel_loss(gen_frames=pred_output, gt_frames=input_gt)

    if lam_two_stream != 0  and use_rec_loss_in_test != 0:
        reconstr_loss = compute_reconstr_loss(input, reconstr_output)
    else:
        reconstr_loss = tf.constant(0.0, dtype=tf.float32)
    
    if adversarial:
        # with tf.variable_scope('discriminator', reuse=None):
        with tf.variable_scope('discriminator'):
            if GAN_type == 'SNGAN':
                discriminator = Discriminator(input_gt, batch_size, is_sn=False)
            elif GAN_type in ['WGAN', 'WGAN-GP']:
                discriminator = Discriminator(input_gt, batch_size,
                                              use_sigmoid=False)
            else:
                discriminator = Discriminator(input_gt, batch_size)
            real_outputs = discriminator.outputs
        # 将参数reuse设置为True时，tf.get_variable 将只能获取已经创建过的变量。
        with tf.variable_scope('discriminator', reuse=True):
            if GAN_type == 'SNGAN':
                discriminator = Discriminator(pred_output, batch_size, is_sn=False)
            elif GAN_type in ['WGAN', 'WGAN-GP']:
                discriminator = Discriminator(pred_output, batch_size,
                                              use_sigmoid=False)
            else:
                discriminator = Discriminator(pred_output, batch_size)
            fake_outputs = discriminator.outputs

        if GAN_type == 'LSGAN':
            # LSGAN, paper: Least Squares Generative Adversarial Networks
            # adv_G_loss
            adv_loss = tf.reduce_mean(tf.square(fake_outputs - 1) / 2)
            # adv_D_loss
            dis_loss = tf.reduce_mean(
                tf.square(real_outputs - 1) / 2) + tf.reduce_mean(
                tf.square(fake_outputs) / 2)
        elif GAN_type == 'WGAN':
            # WGAN, paper: Wasserstein GAN
            # adv_G_loss
            adv_loss = -tf.reduce_mean(fake_outputs)
            # adv_D_loss, 负值
            dis_loss = -tf.reduce_mean(real_outputs - fake_outputs)
        elif GAN_type == 'WGAN-GP':
            # WGAN-GP, paper: Improved Training of Wasserstein GANs
            # gradient penalty 相较 weight penalty则可以让梯度在后向传播的过程中保持平稳
            lambda_gp = 1
            e = tf.random_uniform([batch_size, 1, 1, 1], 0, 1)
            x_hat = e * input_gt + (1 - e) * pred_output

            with tf.variable_scope('discriminator', reuse=True):
                discriminator = Discriminator(x_hat, batch_size)
                grad = tf.gradients(discriminator.outputs, x_hat)[0]

            gradient_penalty = tf.reduce_mean(tf.square(
                tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) - 1))

            dis_loss = tf.reduce_mean(fake_outputs - real_outputs) \
                        + lambda_gp * gradient_penalty

            adv_loss = -tf.reduce_mean(fake_outputs)
        elif GAN_type == 'SNGAN':
            # SNGAN, paper: SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS
            dis_loss = - (tf.reduce_mean(tf.log(real_outputs + epsilon) + tf.log(
                1 - fake_outputs + epsilon)))
            adv_loss = - tf.reduce_mean(tf.log(fake_outputs + epsilon))
        elif GAN_type == 'GAN':
            dis_loss = - (tf.reduce_mean(tf.log(real_outputs) + tf.log(
                1 - fake_outputs)))
            adv_loss = - tf.reduce_mean(tf.log(fake_outputs))
        else:
            print('[!!!] wrong GAN_TYPE')
    else:
        adv_loss = tf.constant(0.0, dtype=tf.float32)
        dis_loss = tf.constant(0.0, dtype=tf.float32)
        

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    if for_remote != 1:
        # fraction of overall amount of memory that each GPU should be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
    
    with tf.Session(config=config) as sess:
        init_or_restore(sess, saver_for_best, best_model_ckpt_dir, logger)

        start_time = time.time()
        try:
            assert os.path.isfile(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'))
            assert os.path.isfile(os.path.join(regularity_score_dir, 'video_length_list.txt'))
        except:
            print('[!!!] 未发现video_length_list.txt 或 ignored_frames_list.txt， 开始计算')
            logger.info('[!!!] 未发现video_length_list.txt 或 ignored_frames_list.txt， 开始计算')
            # total_frame_nums = 0
            IGNORED_FRAMES_LIST = []
            VIDEO_LENGTH_LIST = []
            for i in range(video_nums):
                npy_name = 'testing_frames_{:02d}.npy'.format(i + 1)
                if use_rgb == 1:
                    npy_name = 'rgb_' + npy_name
                data_frames = np.load(os.path.join(npy_dir, npy_name))
                frame_nums = data_frames.shape[0]
                used_frame_nums = frame_nums
                IGNORED_FRAMES_LIST.append(cfg.clip_length-1)
                VIDEO_LENGTH_LIST.append(used_frame_nums)
                # total_frame_nums +=used_frame_nums
            np.savetxt(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'),
                       np.array(IGNORED_FRAMES_LIST), fmt='%d')
            np.savetxt(os.path.join(regularity_score_dir, 'video_length_list.txt'),
                       np.array(VIDEO_LENGTH_LIST), fmt='%d')

        for i in range(video_nums):
            npy_name = 'testing_frames_{:02d}.npy'.format(i + 1)
            if use_rgb == 1:
                npy_name = 'rgb_' + npy_name
            data_frames = np.load(os.path.join(npy_dir, npy_name))
            # frame_nums = data_frames.shape[0]
            # used_frame_nums = frame_nums
            # IGNORED_FRAMES_LIST.append(cfg.clip_length-1)
            # VIDEO_LENGTH_LIST.append(used_frame_nums)
            # total_frame_nums +=used_frame_nums
            score_file = [x for x in os.listdir(regularity_score_dir)
                          if x.startswith('scores_') == True]
            if (len(score_file) != video_nums):

                mse_errors_l = []
                psnr_l = []
                if lam_two_stream != 0  and use_rec_loss_in_test != 0:
                    reconstr_loss_l = []
                    psnr_rec_l = []
                    psnr_half_rec_l = []
                adv_loss_l = []
                psnr_adv_l = []
                psnr_half_adv_l = []
                if generate_heatmap == 1:
                    pixel_loss_l = []

                for j in range(len(data_frames)-cfg.clip_length+1):
                    # [clip_length, h, w]
                    tested_data = data_frames[j:j + cfg.clip_length]
                    # [n, clip_length, h, w] for gray or [n, clip_length, h, w, c] for rgb
                    tested_data = np.expand_dims(tested_data, axis=0)
                    if use_rgb == 0:
                        # [n, clip_length, h, w, 1]
                        tested_data = np.expand_dims(tested_data, axis=-1)

                    pixel_loss_v, adv_loss_v, reconstr_loss_v, mse_error_v, psnr_v = sess.run(
                            [pixel_loss, adv_loss, reconstr_loss, mse_error, psnr],
                            feed_dict={data: tested_data,
                                       batch_size:1})

                    if generate_heatmap == 1:
                        pixel_loss_l.append(pixel_loss_v)
                    mse_errors_l.append(mse_error_v)
                    psnr_l.append(psnr_v)
                    adv_loss_l.append(adv_loss_v)
                    psnr_adv_l.append(-psnr_v + adv_loss_v)
                    psnr_half_adv_l.append(-psnr_v + 0.5*adv_loss_v)
                    if lam_two_stream != 0  and use_rec_loss_in_test != 0:
                        reconstr_loss_l.append(reconstr_loss_v)
                        psnr_rec_l.append(-psnr_v + reconstr_loss_v)
                        psnr_half_rec_l.append(-psnr_v + 0.5 * reconstr_loss_v)

                if generate_heatmap == 1:
                    pixel_losses = np.array(pixel_loss_l)
                mse_errors = np.array(mse_errors_l)
                psnrs = np.array(psnr_l)
                adv_losses = np.array(adv_loss_l)
                if lam_two_stream != 0  and use_rec_loss_in_test != 0:
                    reconstr_losses = np.array(reconstr_loss_l)
                    psnr_recs = np.array(psnr_rec_l)
                    psnr_half_recs = np.array(psnr_half_rec_l)
                psnr_advs = np.array(psnr_adv_l)
                psnr_half_advs = np.array(psnr_half_adv_l)

                lam_s = (max(psnrs) / max(adv_losses)) / 10
                # max_psnrs = max(psnrs)
                # max_adv_losses = max(adv_losses)
                # print('---- max_psnrs:{}'.format(max_psnrs))
                # logger.info('---- max_psnrs:{}'.format(max_psnrs))
                # print('---- max_adv_losses:{}'.format(max_adv_losses))
                # logger.info('---- max_adv_losses:{}'.format(max_adv_losses))
                
                psnr_combine_losses = -psnrs + lam_s * adv_losses
                psnr_combine_losses_1 = -psnrs + 50 * adv_losses
                psnr_combine_losses_2 = -psnrs + 10 * adv_losses

                if generate_heatmap == 1:
                    save_pixel_loss(pixel_losses, regularity_score_dir, i, 'pixel_loss')
                compute_and_save_scores(mse_errors, regularity_score_dir, i, 'mse')
                compute_and_save_scores(psnrs, regularity_score_dir, i, 'psnr')
                compute_and_save_scores(adv_losses, regularity_score_dir, i, 'adv')
                compute_and_save_scores(psnr_advs, regularity_score_dir, i, 'psnr_adv')
                compute_and_save_scores(psnr_half_advs, regularity_score_dir, i, 'psnr_half_adv')
                if lam_two_stream != 0  and use_rec_loss_in_test != 0:
                    compute_and_save_scores(reconstr_losses, regularity_score_dir, i, 'rec')
                    compute_and_save_scores(psnr_recs, regularity_score_dir, i, 'psnr_rec')
                    compute_and_save_scores(psnr_half_recs, regularity_score_dir, i,
                                            'psnr_half_rec')
                
                # combine D and G
                compute_and_save_scores(psnr_combine_losses, regularity_score_dir, i, 
                                        'psnr_combine_losses')
                compute_and_save_scores(psnr_combine_losses_1, regularity_score_dir, i,
                                        'psnr_combine_losses_1')
                compute_and_save_scores(psnr_combine_losses_2, regularity_score_dir, i,
                                        'psnr_combine_losses_2')


    print('AUC and EER result:')
    logger.info('AUC and EER result:')
    # compute auc and eer
    for error_name in ['mse', 'psnr', 'adv', 'psnr_adv', 'psnr_half_adv', 'psnr_combine_losses',
                       'psnr_combine_losses_1', 'psnr_combine_losses_2']:
        assert os.path.exists(os.path.join(regularity_score_dir, error_name)) is True\
            , '[!!!] error_name:{} is non-existent.'.format(error_name)
        print('---- error_name:{}'.format(error_name))
        logger.info('---- error_name:{}'.format(error_name))
        auc = compute_auc(video_nums, regularity_score_dir, error_name, args.dataset,
                          gt_root_dir)
        print('auc = {:09f}'.format(auc))
        logger.info('auc = {:09f}'.format(auc))
        eer = compute_eer(video_nums, regularity_score_dir, error_name, args.dataset,
                          gt_root_dir)
        print('eer = {:09f}'.format(eer))
        logger.info('eer = {:09f}'.format(eer))

        # plot score
        plot_score(video_nums, args.dataset, regularity_score_dir, error_name, logger,
                   gt_root_dir, cfg.clip_length - 1)
    if lam_two_stream != 0 and use_rec_loss_in_test != 0:
        # compute auc and eer
        for error_name in ['rec', 'psnr_rec', 'psnr_half_rec']:
            assert os.path.exists(os.path.join(regularity_score_dir, error_name)) is True\
                , '[!!!] error_name:{} is non-existent.'.format(error_name)
            print('---- error_name:{}'.format(error_name))
            logger.info('---- error_name:{}'.format(error_name))
            auc = compute_auc(video_nums, regularity_score_dir, error_name, args.dataset,
                              gt_root_dir)
            print('auc = {:09f}'.format(auc))
            logger.info('auc = {:09f}'.format(auc))
            eer = compute_eer(video_nums, regularity_score_dir, error_name, args.dataset,
                              gt_root_dir)
            print('eer = {:09f}'.format(eer))
            logger.info('eer = {:09f}'.format(eer))

            # plot score
            plot_score(video_nums, args.dataset, regularity_score_dir, error_name, logger,
                       gt_root_dir, cfg.clip_length - 1)

    # plot heatmap
    if generate_heatmap == 1:
        for error_name in ['pixel_loss']:
            assert os.path.exists(os.path.join(regularity_score_dir, error_name)) is True\
                , '[!!!] error_name:{} is non-existent.'.format(error_name)
            print('---- error_name:{}'.format(error_name))
            logger.info('---- error_name:{}'.format(error_name))

            # plot score
            plot_heatmap(video_nums, args.dataset, regularity_score_dir, error_name, logger,
                         cfg.clip_length - 1, dataset_root_dir, cfg, gt_root_dir)
        print('[!!!] model name:{}'.format(model_name))
        logger.info('[!!!] model name:{}'.format(model_name))

"""
Test_in_train phase
"""
def test_in_train(cfg, logger, sess, model_name):
    pass


def main(*argc, **argv):
    cfg = Config()

    model_name = '{}_{}_{}_{}_{}'.format(args.dataset, cfg.batch_size, cfg.opt, cfg.lr, cfg.epochs)
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
