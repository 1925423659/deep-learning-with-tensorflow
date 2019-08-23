import tensorflow

g_1 = tensorflow.Graph()
with g_1.as_default():
    v = tensorflow.get_variable('v', [1, 2], initializer=tensorflow.zeros_initializer())

g_2 = tensorflow.Graph()
with g_2.as_default():
    v = tensorflow.get_variable('v', [2, 1], initializer=tensorflow.ones_initializer())

with tensorflow.Session(graph=g_1) as sess:
    tensorflow.global_variables_initializer().run()
    with tensorflow.variable_scope('', reuse=True):
        print(sess.run(tensorflow.get_variable('v')))

with tensorflow.Session(graph=g_2) as sess:
    tensorflow.global_variables_initializer().run()
    with tensorflow.variable_scope('', reuse=True):
        print(sess.run(tensorflow.get_variable('v')))