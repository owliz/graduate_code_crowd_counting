import tensorflow as tf
import os
import numpy as np
import random
img_rows = 512
img_cols = 512
fac = 8

# def _parse_example(self, serial_exmp):
#     # tf.FixedLenFeature([], tf.string)  # 0D, 标量
#     # tf.FixedLenFeature([3]volume...)   1D，长度为3
#     features = {'volume': tf.FixedLenFeature([], tf.string)}
#     parsed_features = tf.parse_single_example(serial_exmp, features)
#     volume = tf.decode_raw(parsed_features['volume'], tf.float32)
#     return volume

def _parse_example(recordfile):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(recordfile, feature)
    image = tf.reshape(tf.decode_raw(features['train/image'], tf.float32),[224,224,3])
    image = image - tf.constant([103.99, 116.779, 123.68])
    image = tf.image.resize_images(image, [img_rows, img_cols])
    label = tf.reshape(tf.decode_raw(features['train/label'], tf.float32),[224,224,1])
    label = tf.image.resize_images(label, [img_rows//fac, img_cols//fac])
    return image,label


def _corrupt_brightness(image, mask):
    """Radnomly applies a random brightness change."""
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, mask


def _corrupt_contrast(image, mask):
    """Randomly applies a random contrast change."""
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _corrupt_saturation(image, mask):
    """Randomly applies a random saturation change."""
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _flip_left_right(image, mask):
    """Randomly flips image and mask left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)

    return image, mask


def _crop_random(image, mask):
    """Randomly crops image and mask in accord."""
    seed = random.random()
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [int(img_rows * 0.85), int(img_cols * 0.85), 3], seed=seed), lambda: tf.identity(image))
    mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        mask, [int(img_rows//fac * 0.85), int(img_cols//fac * 0.85), 1], seed=seed),
                   lambda: tf.identity(mask))
    image = tf.expand_dims(image, axis=0)
    mask = tf.expand_dims(mask, axis=0)

    image = tf.image.resize_images(image, [img_rows, img_cols])
    mask = tf.image.resize_images(mask, [img_rows//fac, img_cols//fac])

    image = tf.squeeze(image, axis=0)
    mask = tf.squeeze(mask, axis=0)

    return image, mask


def data_iterator(dataset, phase, cfg, batch_size, tfrecord_root_dir, logger, num_threads=2,
                  prefetch=30):
        # train_volume_num = 0
        dir_path = os.path.join(tfrecord_root_dir, dataset)
        if phase == 'train':
            file_dir = os.path.join(dir_path, 'train_data.tfrecords')
        elif phase == 'test':
            file_dir = os.path.join(dir_path, 'test_data.tfrecords')
        
        dataset = tf.data.TFRecordDataset(file_dir)
        dataset = dataset.map(_parse_example)

        # if cfg.quick_train == 1:
        #     """
        #     for choose hyper-parameter quickly
        #     """
        #     percent = 0.2
        #     small_train_num = int(train_volume_num * percent)
        #     print('[!!!] {} of train set is used for quick training.'.format(small_train_num,
        #                                                                     percent))
        #     logger.info('[!!!] {} of train set is used for quick training.'.format(
        #         small_train_num, percent))
        #     dataset = dataset.take(small_train_num)

        if phase == 'train' and cfg.augment == 1:
            dataset = dataset.map(_corrupt_brightness,
                                              num_parallel_calls=num_threads).prefetch(prefetch)

            dataset = dataset.map(_corrupt_contrast,
                                              num_parallel_calls=num_threads).prefetch(prefetch)

            dataset = dataset.map(_corrupt_saturation,
                                              num_parallel_calls=num_threads).prefetch(prefetch)

            dataset = dataset.map(_crop_random,
                                              num_parallel_calls=num_threads).prefetch(prefetch)

            dataset = dataset.map(_flip_left_right,
                                              num_parallel_calls=num_threads).prefetch(prefetch)

        if phase == 'train':
            dataset = dataset.shuffle(buffer_size=prefetch)
        # drop reminder data with insufficient batch size
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset.make_initializable_iterator()








