import tensorflow

v = tensorflow.Variable(0, dtype=tensorflow.float32, name='v')
ema = tensorflow.train.ExponentialMovingAverage(0.99)
saver = tensorflow.train.Saver(ema.variables_to_restore())

with tensorflow.Session() as session:
    saver.restore(session, 'dataset/model/model.ckpt')
    print(session.run(v))