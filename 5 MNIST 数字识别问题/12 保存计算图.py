import tensorflow
from tensorflow.python.framework import graph_util

v_1 = tensorflow.Variable(tensorflow.constant(1.0, shape=[1]), name='v_1')
v_2 = tensorflow.Variable(tensorflow.constant(2.0, shape=[1]), name='v_2')
result = v_1 + v_2

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    graph_def = tensorflow.get_default_graph().as_graph_def()

    output_graph_def = graph_util.convert_variables_to_constants(session, graph_def, ['add'])

    with tensorflow.gfile.GFile('dataset/model/combined_model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())