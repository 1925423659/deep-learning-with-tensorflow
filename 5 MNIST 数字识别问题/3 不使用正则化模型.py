from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/MNIST', one_hot=True)

import tensorflow

x = tensorflow.placeholder(tensorflow.float32, [None, 784], 'x-input')
y = tensorflow.placeholder(tensorflow.float32, [None, 10], 'y-input')

weight_1 = tensorflow.Variable(tensorflow.truncated_normal([784, 500], stddev=0.1))
bias_1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[500]))
weight_2 = tensorflow.Variable(tensorflow.truncated_normal([500, 10], stddev=0.1))
bias_2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[10]))

layer = tensorflow.nn.relu(tensorflow.matmul(x, weight_1) + bias_1)
z = tensorflow.matmul(layer, weight_2) + bias_2

global_step = tensorflow.Variable(0, trainable=False)
ema = tensorflow.train.ExponentialMovingAverage(0.99, global_step)
ema_op = ema.apply(tensorflow.trainable_variables())

ema_layer = tensorflow.nn.relu(tensorflow.matmul(x, ema.average(weight_1)) + bias_1)
ema_z = tensorflow.matmul(ema_layer, ema.average(weight_2)) + bias_2

cross_entropy = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=tensorflow.argmax(y, 1))
cross_entropy_mean = tensorflow.reduce_mean(cross_entropy)

loss = cross_entropy_mean

learning_rate = tensorflow.train.exponential_decay(0.8, global_step, mnist.train.num_examples / 100, 0.99)

train_step = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

with tensorflow.control_dependencies([train_step, ema_op]):
    train_op = tensorflow.no_op('train')

correct_prediction = tensorflow.equal(tensorflow.argmax(ema_z, 1), tensorflow.argmax(y, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

with tensorflow.Session() as session:
    tensorflow.global_variables_initializer().run()

    validation_dict = {x: mnist.validation.images, y: mnist.validation.labels}
    test_dict = {x: mnist.test.images, y: mnist.test.labels}

    for i in range(30000):
        if i % 1000 == 0:
            validation_accuracy_result = session.run(accuracy, feed_dict=validation_dict)
            print('%d validation accuracy %g' % (i, validation_accuracy_result))
        
        xs, ys = mnist.train.next_batch(100)
        session.run(train_op, feed_dict={x: xs, y: ys})
    
    test_accuracy_result = session.run(accuracy, feed_dict=test_dict)
    print('test accuracy %g' % (test_accuracy_result))