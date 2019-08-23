import tensorflow

reader = tensorflow.train.NewCheckpointReader('dataset/model/model.ckpt')

variable_map = reader.get_variable_to_shape_map()
for variable_name in variable_map:
    print(variable_name, variable_map[variable_name])

print('v_1 is', reader.get_tensor('v_1'))