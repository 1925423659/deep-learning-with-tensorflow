import tensorflow
import numpy
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/MNIST', dtype=tensorflow.uint8, one_hot=True)

with tensorflow.python_io.TFRecordWriter('dataset/TFRecord/train.tfrecord') as writer:
    for i in range(mnist.train.num_examples):
        image = mnist.train.images[i]
        label = mnist.train.labels[i]
        pixel_int64_list = tensorflow.train.Int64List(value=[image.shape[0]])
        label_int64_list = tensorflow.train.Int64List(value=[numpy.argmax(label)])
        image_bytes_list = tensorflow.train.BytesList(value=[image.tostring()])
        feature = {'pixel': tensorflow.train.Feature(int64_list=pixel_int64_list),
                'label': tensorflow.train.Feature(int64_list=label_int64_list),
                'image': tensorflow.train.Feature(bytes_list=image_bytes_list)}
        features = tensorflow.train.Features(feature=feature)
        example = tensorflow.train.Example(features=features)
        writer.write(example.SerializeToString())
writer.close()