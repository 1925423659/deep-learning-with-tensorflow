import tensorflow

v = tensorflow.Variable(0, dtype=tensorflow.float32, name='v')

with tensorflow.Session() as session:
    saver = tensorflow.train.Saver({'v/ExponentialMovingAverage': v})
    saver.restore(session, 'dataset/model/model.ckpt')
    print(session.run(v))