import tensorflow

random_1 = tensorflow.random_normal((2, 3), stddev=1, seed=1)
random_2 = tensorflow.random_normal((3, 1), stddev=1, seed=1)
weight_1 = tensorflow.Variable(random_1)
weight_2 = tensorflow.Variable(random_2)

x = tensorflow.constant([[0.7, 0.9]])

a = tensorflow.matmul(x, weight_1)
y = tensorflow.matmul(a, weight_2)

session = tensorflow.Session()
session.run(weight_1.initializer)
session.run(weight_2.initializer)
print(session.run(x))
print(session.run(weight_1))
print(session.run(a))
print(session.run(weight_2))
print(session.run(y))
session.close()