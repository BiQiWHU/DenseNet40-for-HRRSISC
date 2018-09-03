
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import cv2


# In[2]:


test_ratio = 5
crop_size = 227
scale_size = 256
n_classes = 21


# In[3]:


def get_records(dataset_path, ext=".tif"):
    writer_train = tf.python_io.TFRecordWriter("train.tfrecords")
    writer_test = tf.python_io.TFRecordWriter("test.tfrecords")
    class_names = [f for f in os.listdir(dataset_path) if not f.startswith('.')]

    for index, name in enumerate(class_names):
        print(index, ",", name)
        directory = os.path.join(dataset_path, name)
        class_image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]

        for i, img_path in enumerate(class_image_paths):
            img = cv2.imread(img_path)
            ### for AlexNet  for DenseNet on ImageNet
            # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            ### for DenseNet on CIFAR   DenseNet40
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # 8 2
            if (i + 1) % test_ratio == 2:
                writer_test.write(example.SerializeToString())
            else:
                writer_train.write(example.SerializeToString())

    writer_train.close()
    writer_test.close()


# In[4]:


def read_and_decode(filename, distort_images=True):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    ### for Alex and Dense on ImageNet ,reshape to 256  For each pixel,in DenseNet we need to normalize it to [0,1]
    # img = tf.reshape(img, [256, 256, 3])
    ### for Dense on CIFAR, reshape to 32 
    img = tf.reshape(img, [32, 32, 3])/255
    label = tf.cast(features['label'], tf.int32)

    if distort_images:
        # Randomly flip the image horizontally or vertically, change the brightness and contrast
        ##  for AlexNet and DenseNet on ImageNet   random crop
        ##  for DenseNet on CIFAR  do not radom crop
        # img = tf.random_crop(img, [227, 227, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)

    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label


# In[5]:


def input_pipeline(filename, batch_size,is_shuffle=True,is_train=True):
    example, label = read_and_decode(filename)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    if is_shuffle:
        example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

    return example_batch, label_batch


# In[6]:


if __name__ == '__main__':
    get_records("UCMerced_LandUse/Images")
    # 0, agricultural
    # 1, airplane
    # 2, baseballdiamond
    # 3, beach
    # 4, buildings
    # 5, chaparral
    # 6, denseresidential
    # 7, forest
    # 8, freeway
    # 9, golfcourse
    # 10, harbor
    # 11, intersection
    # 12, mediumresidential
    # 13, mobilehomepark
    # 14, overpass
    # 15, parkinglot
    # 16, river
    # 17, runway
    # 18, sparseresidential
    # 19, storagetanks
    # 20, tenniscourt

