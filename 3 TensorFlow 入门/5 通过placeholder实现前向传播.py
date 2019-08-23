import tensorflow

random_1 = tensorflow.random_normal([2, 3], stddev=1)
random_2 = tensorflow.random_normal([3, 1], stddev=1)

weight_1 = tensorflow.Variable(random_1)
weight_2 = tensorflow.Variable(random_2)

x = tensorflow.placeholder(tensorflow.float32, shape=[1, 2], name='input')

a = tensorflow.matmul(x, weight_1)
y = tensorflow.matmul(a, weight_2)

session = tensorflow.Session()

variables_initializer = tensorflow.global_variables_initializer()
session.run(variables_initializer)

feed_dict = {x: [[0.7, 0.9]]}
result = session.run(y, feed_dict=feed_dict)
print(result)