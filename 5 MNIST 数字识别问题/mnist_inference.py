import tensorflow

def get_weight_variable(shape, regularizer):
    weight = tensorflow.get_variable('weight', shape, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    
    if regularizer != None:
        tensorflow.add_to_collection('loss', regularizer(weight))
    
    return weight

def inference(input_tensor, regularizer):
    with tensorflow.variable_scope('layer_1'):
        weight = get_weight_variable([784, 500], regularizer)
        bias = tensorflow.get_variable('bias', [500], initializer=tensorflow.constant_initializer(0))
        layer_1 = tensorflow.nn.relu(tensorflow.matmul(input_tensor, weight) + bias)
    
    with tensorflow.variable_scope('layer_2'):
        weight = get_weight_variable([500, 10], regularizer)
        bias = tensorflow.get_variable('bias', [10], initializer=tensorflow.constant_initializer(0))
        layer_2 = tensorflow.matmul(layer_1, weight) + bias
    
    return layer_2