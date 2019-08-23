import tensorflow

var = tensorflow.Variable(0, dtype=tensorflow.float32)
step = tensorflow.Variable(0, trainable=False)

ema = tensorflow.train.ExponentialMovingAverage(0.99, step)
maintain_averages_op = ema.apply([var])

with tensorflow.Session() as session:
    init_op = tensorflow.global_variables_initializer()
    session.run(init_op)

    print(session.run([var, ema.average(var)]))

    session.run(tensorflow.assign(var, 5))
    session.run(maintain_averages_op)
    print(session.run([var, ema.average(var)]))

    session.run(tensorflow.assign(step, 10000))
    session.run(tensorflow.assign(var, 10))
    session.run(maintain_averages_op)
    print(session.run([var, ema.average(var)]))

    session.run(maintain_averages_op)
    print(session.run([var, ema.average(var)]))