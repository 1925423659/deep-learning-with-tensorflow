import tensorflow

v_1 = tensorflow.Variable(tensorflow.constant(1.0, shape=[1]), name='v_1')
v_2 = tensorflow.Variable(tensorflow.constant(2.0, shape=[1]), name='v_2')
result = v_1 + v_2

saver = tensorflow.train.Saver()
saver.export_meta_graph('dataset/model/model.ckpt.meta.json', as_text=True)