import tensorflow

def inference(input_tensor, regularizer, train):
    with tensorflow.variable_scope('conv_1'):
        conv_1_weight = tensorflow.get_variable('weight', [5, 5, 1, 32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        conv_1_bias = tensorflow.get_variable('bias', [32], initializer=tensorflow.constant_initializer(0))
        conv_1 = tensorflow.nn.conv2d(input_tensor, conv_1_weight, [1, 1, 1, 1], 'SAME')
        conv_1_features = tensorflow.nn.bias_add(conv_1, conv_1_bias)
        conv_1_relu = tensorflow.nn.relu(conv_1_features)
    
    with tensorflow.name_scope('pool_1'):
        pool_1 = tensorflow.nn.max_pool(conv_1_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    with tensorflow.variable_scope('conv_2'):
        conv_2_weight = tensorflow.get_variable('weight', [5, 5, 32, 64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        conv_2_bias = tensorflow.get_variable('bias', [64], initializer=tensorflow.constant_initializer(0))
        conv_2 = tensorflow.nn.conv2d(pool_1, conv_2_weight, [1, 1, 1, 1], 'SAME')
        conv_2_features = tensorflow.nn.bias_add(conv_2, conv_2_bias)
        conv_2_relu = tensorflow.nn.relu(conv_2_features)
    
    with tensorflow.name_scope('pool_2'):
        pool_2 = tensorflow.nn.max_pool(conv_2_relu, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    with tensorflow.name_scope('reshape'):
        pool_2_shape = pool_2.get_shape().as_list()
        HWC = pool_2_shape[1] * pool_2_shape[2] * pool_2_shape[3]
        pool_2_reshape = tensorflow.reshape(pool_2, [pool_2_shape[0], HWC])
    
    with tensorflow.variable_scope('fc_1'):
        fc_1_weight = tensorflow.get_variable('weight', [HWC, 512], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tensorflow.add_to_collection('loss', regularizer(fc_1_weight))
        fc_1_bias = tensorflow.get_variable('bias', [512], initializer=tensorflow.constant_initializer(0.1))
        fc_1_features = tensorflow.matmul(pool_2_reshape, fc_1_weight) + fc_1_bias
        fc_1 = tensorflow.nn.relu(fc_1_features)
        if train:
            fc_1 = tensorflow.nn.dropout(fc_1, 0.5)
    
    with tensorflow.variable_scope('fc_2'):
        fc_2_weight = tensorflow.get_variable('weight', [512, 10], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tensorflow.add_to_collection('loss', regularizer(fc_2_weight))
        fc_2_bias = tensorflow.get_variable('bias', [10], initializer=tensorflow.constant_initializer(0.1))
        fc_2_features = tensorflow.matmul(fc_1, fc_2_weight) + fc_2_bias
    
    return fc_2_features