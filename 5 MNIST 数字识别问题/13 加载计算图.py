import tensorflow
from tensorflow.python.platform import gfile

with tensorflow.Session() as session:
    with gfile.FastGFile('dataset/model/combined_model.pb', 'rb') as f:
        graph_def = tensorflow.GraphDef()
        graph_def.ParseFromString(f.read())
    
    result = tensorflow.import_graph_def(graph_def, return_elements=['add:0'])
    print(session.run(result))