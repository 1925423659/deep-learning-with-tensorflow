import tensorflow

reader = tensorflow.TFRecordReader()
filename_queue = tensorflow.train.string_input_producer(['dataset/TFRecord/train.tfrecord'])
_, serialized_example = reader.read(filename_queue)

features = {
    'image': tensorflow.FixedLenFeature([], tensorflow.string),
    'pixel': tensorflow.FixedLenFeature([], tensorflow.int64),
    'label': tensorflow.FixedLenFeature([], tensorflow.int64)
}
features = tensorflow.parse_single_example(serialized_example, features=features)

image = tensorflow.decode_raw(features['image'], tensorflow.uint8)
label = tensorflow.cast(features['label'], tensorflow.int32)
pixel = tensorflow.cast(features['pixel'], tensorflow.int32)

sess = tensorflow.Session()
coord = tensorflow.train.Coordinator()
threads = tensorflow.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    result = sess.run([image, label, pixel])
    print(result)