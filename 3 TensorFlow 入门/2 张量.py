import tensorflow

a = tensorflow.constant([1.0, 2.0], name='a')
b = tensorflow.constant([2.0, 3.0], name='b')
result = tensorflow.add(a, b, name='add')
print(result)
print(tensorflow.Session().run(result))
