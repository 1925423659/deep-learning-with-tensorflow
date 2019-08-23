from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('dataset/MNIST/', one_hot=True)

print('mnist train num examples', mnist.train.num_examples)

print('mnist validation num examples', mnist.validation.num_examples)

print('mnist test num examples', mnist.test.num_examples)

print('mnist train images', mnist.train.images[0])

print('mnist train labels', mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print('x shape', xs.shape)
print('y shape', ys.shape)