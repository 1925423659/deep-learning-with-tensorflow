import tensorflow

# with tensorflow.variable_scope('foo'):
#     v = tensorflow.get_variable('v', [1], initializer=tensorflow.constant_initializer(1))

# with tensorflow.variable_scope('foo'):
#     v = tensorflow.get_variable('v', [1])

# with tensorflow.variable_scope('foo', reuse=True):
#     v_1 = tensorflow.get_variable('v', [1])
#     print(v == v_1)

# with tensorflow.variable_scope('bar', reuse=True):
#     v = tensorflow.get_variable('v', [1])


# with tensorflow.variable_scope('root'):
#     print(tensorflow.get_variable_scope().reuse)

#     with tensorflow.variable_scope('foo', reuse=True):
#         print(tensorflow.get_variable_scope().reuse)

#         with tensorflow.variable_scope('bar'):
#             print(tensorflow.get_variable_scope().reuse)
    
#     print(tensorflow.get_variable_scope().reuse)


v_1 = tensorflow.get_variable('v', [1])
print(v_1.name)

with tensorflow.variable_scope('foo'):
    v_2 = tensorflow.get_variable('v', [1])
    print(v_2.name)

with tensorflow.variable_scope('foo'):
    with tensorflow.variable_scope('bar'):
        v_3 = tensorflow.get_variable('v', [1])
        print(v_3.name)
    
    v_4 = tensorflow.get_variable('v_1', [1])
    print(v_4.name)

with tensorflow.variable_scope('', reuse=True):
    v_5 = tensorflow.get_variable('foo/bar/v', [1])
    print(v_5 == v_3)

    v_6 = tensorflow.get_variable('foo/v_1', [1])
    print(v_6 == v_4)