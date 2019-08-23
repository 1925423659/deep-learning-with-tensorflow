import tensorflow

v_1 = tensorflow.Variable(tensorflow.constant(1.0, tensorflow.float32, [1]), name='v_1')
v_2 = tensorflow.Variable(tensorflow.constant(2.0, tensorflow.float32, [1]), name='v_2')
result = v_1 + v_2

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    saver = tensorflow.train.Saver()
    saver.save(session, 'dataset/model/model.ckpt')