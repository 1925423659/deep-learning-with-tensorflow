import tensorflow


v_1 = tensorflow.Variable(tensorflow.constant(0, tensorflow.float32, [1]), name='v_1')
v_2 = tensorflow.Variable(tensorflow.constant(0, tensorflow.float32, [1]), name='v_2')
result = v_1 + v_2

with tensorflow.Session() as session:
    saver = tensorflow.train.Saver()
    saver.restore(session, 'dataset/model/model.ckpt')
    print(session.run(result))