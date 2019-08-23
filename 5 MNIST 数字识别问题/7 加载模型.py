import tensorflow

saver = tensorflow.train.import_meta_graph('dataset/model/model.ckpt.meta')

with tensorflow.Session() as session:
    saver.restore(session, 'dataset/model/model.ckpt')
    result = tensorflow.get_default_graph().get_tensor_by_name('add:0')
    print(session.run(result))
