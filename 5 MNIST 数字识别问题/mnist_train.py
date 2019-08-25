import os
import tensorflow
import mnist_inference

from tensorflow.examples.tutorials.mnist import input_data

def train(mnist):
    x = tensorflow.placeholder(tensorflow.float32, [None, 784], 'x-input')
    y = tensorflow.placeholder(tensorflow.float32, [None, 10], 'y-input')

    regularizer = tensorflow.contrib.layers.l2_regularizer(0.0001)
    z = mnist_inference.inference(x, regularizer)
    
    global_step = tensorflow.Variable(0, trainable=False)

    ema = tensorflow.train.ExponentialMovingAverage(0.99, global_step)
    ema_op = ema.apply(tensorflow.trainable_variables())

    cross_entropy = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=tensorflow.argmax(y, 1))
    cross_entropy_mean = tensorflow.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tensorflow.add_n(tensorflow.get_collection('loss'))

    learning_rate = tensorflow.train.exponential_decay(0.8, global_step, mnist.train.num_examples / 100, 0.99)
    
    train_step = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tensorflow.control_dependencies([train_step, ema_op]):
        train_op = tensorflow.no_op('train')
    
    with tensorflow.Session() as session:
        tensorflow.global_variables_initializer().run()

        saver = tensorflow.train.Saver()

        for i in range(30):
            for j in range(1000):
                xs, ys = mnist.train.next_batch(100)
                _, loss_value, step_value = session.run([train_op, loss, global_step], feed_dict={x: xs, y: ys})
            
            print('%d, %g' % (step_value, loss_value))
            saver.save(session, 'dataset/model/model.ckpt', global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('dataset/MNIST', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tensorflow.app.run()