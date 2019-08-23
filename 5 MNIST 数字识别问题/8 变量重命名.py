import tensorflow

v_1 = tensorflow.Variable(tensorflow.constant(1.0, shape=[1]), name='other-v_1')
v_2 = tensorflow.Variable(tensorflow.constant(2.0, shape=[1]), name='other-v_2')

saver = tensorflow.train.Saver({'v_1': v_1, 'v_2': v_2})