import tensorflow
import numpy
import time
import LeNet5_inference

from tensorflow.examples.tutorials.mnist import input_data

def evaluate(mnist):
    with tensorflow.Graph().as_default() as graph:
        x = tensorflow.placeholder(tensorflow.float32, [mnist.validation.num_examples, 28, 28, 1], 'x-input')
        y = tensorflow.placeholder(tensorflow.float32, [mnist.validation.num_examples, 10], 'y-input')
        z = LeNet5_inference.inference(x, None, False)

        correct_prediction = tensorflow.equal(tensorflow.argmax(z, 1), tensorflow.argmax(y, 1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

        ema = tensorflow.train.ExponentialMovingAverage(0.99)
        variables_to_restore = ema.variables_to_restore()
        saver = tensorflow.train.Saver(variables_to_restore)

        while(True):
            with tensorflow.Session() as session:
                ckpt = tensorflow.train.get_checkpoint_state('dataset/model')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(session, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    x_reshape = numpy.reshape(mnist.validation.images, [mnist.validation.num_examples, 28, 28, 1])
                    accuracy_score = session.run(accuracy, feed_dict={x: x_reshape, y: mnist.validation.labels})
                    print('step %s, validation accuracy %g' % (global_step, accuracy_score))
                else:
                    print('no checkpoint file found')
            time.sleep(10)

def main(argv=None):
    mnist = input_data.read_data_sets('dataset/MNIST', one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tensorflow.app.run()