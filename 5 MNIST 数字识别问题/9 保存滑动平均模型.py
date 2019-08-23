import tensorflow

v = tensorflow.Variable(0, dtype=tensorflow.float32, name='v')
for variable in tensorflow.global_variables():
    print(variable.name)

ema = tensorflow.train.ExponentialMovingAverage(0.99)
ema_op = ema.apply(tensorflow.global_variables())
for variable in tensorflow.global_variables():
    print(variable.name)

saver = tensorflow.train.Saver()
with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    session.run(tensorflow.assign(v, 10))
    session.run(ema_op)

    saver.save(session, 'dataset/model/model.ckpt')
    print(session.run([v, ema.average(v)]))