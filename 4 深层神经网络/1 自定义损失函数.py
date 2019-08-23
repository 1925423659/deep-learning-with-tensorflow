from numpy.random import RandomState

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x_1 + x_2 + (rdm.rand() / 10.0 - 0.05)] for (x_1, x_2) in X]

import tensorflow

batch_size = 8

x = tensorflow.placeholder(tensorflow.float32, [None, 2], 'x-input')
y = tensorflow.placeholder(tensorflow.float32, [None, 1], 'y-input')

random = tensorflow.random_normal([2, 1], stddev=1, seed=1)
weight = tensorflow.Variable(random)

a = tensorflow.matmul(x, weight)

loss_less = 10
loss_more = 1
input_tensor = tensorflow.where(tensorflow.greater(a, y), (a - y) * loss_more, (y - a) * loss_less)
loss = tensorflow.reduce_mean(input_tensor)

train_step = tensorflow.train.AdamOptimizer(0.001).minimize(loss)

with tensorflow.Session() as session:
    init_op = tensorflow.global_variables_initializer()
    session.run(init_op)

    STEPS = 5000
    for i in range(STEPS):
        begin = (i * batch_size) % dataset_size
        end = min(begin + batch_size, dataset_size)

        feed_dict = {x: X[begin: end], y: Y[begin: end]}
        session.run(train_step, feed_dict=feed_dict)
    
    print(session.run(weight))