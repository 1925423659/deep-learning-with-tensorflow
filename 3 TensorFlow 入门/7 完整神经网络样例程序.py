from numpy.random import RandomState

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x_1 + x_2 < 1)] for (x_1, x_2) in X]

import tensorflow

batch_size = 8

random_1 = tensorflow.random_normal([2, 3], stddev=1, seed=1)
random_2 = tensorflow.random_normal([3, 1], stddev=1, seed=1)
weight_1 = tensorflow.Variable(random_1)
weight_2 = tensorflow.Variable(random_2)

x = tensorflow.placeholder(tensorflow.float32, [None, 2], 'x-input')
y = tensorflow.placeholder(tensorflow.float32, [None, 1], 'y-input')

a = tensorflow.matmul(x, weight_1)
b = tensorflow.matmul(a, weight_2)

sigmoid = tensorflow.sigmoid(b)
input_tensor = y * tensorflow.log(tensorflow.clip_by_value(sigmoid, 1e-10, 1)) + (1 - y) * tensorflow.log(tensorflow.clip_by_value(1 - sigmoid, 1e-10, 1))
cross_entropy = -tensorflow.reduce_mean(input_tensor)
train_step = tensorflow.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tensorflow.Session() as session:
    init_op = tensorflow.global_variables_initializer()
    session.run(init_op)

    print(session.run(weight_1))
    print(session.run(weight_2))

    STEPS = 20000
    for i in range(STEPS):
        begin = (i * batch_size) % dataset_size
        end = min(begin + batch_size, dataset_size)

        feed_dict = {x: X[begin: end], y: Y[begin: end]}
        session.run(train_step, feed_dict=feed_dict)

        if i % 100 == 0:
            feed_dict = {x: X, y: Y}
            cross_entropy_result = session.run(cross_entropy, feed_dict=feed_dict)
            print('%d %g' % (i, cross_entropy_result))
    
    print(session.run(weight_1))
    print(session.run(weight_2))