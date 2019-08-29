import tensorflow
import numpy
import os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

def get_tuned_variables():
    exclusions = [scope.strip() for scope in 'InceptionV3/Logits,InceptionV3/AuxLogits'.split(',')]
    variables_to_restore = []

    for variable in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if variable.op.name.startwith(exclusion):
                excluded = True
                break
        
        if not excluded:
            variables_to_restore.append(variable)

    return variables_to_restore

def get_trainable_variables():
    scopes = [scope.strip() for scope in 'InceptionV3/Logits,InceptionV3/AuxLogits'.split(',')]
    variables_to_train = []

    for scope in scopes:
        variables = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    
    return variables_to_train

def main():
    processed_data_train = numpy.load('dataset/flower_photos_train.npy')
    processed_data_validation = numpy.load('dataset/flower_photos_validation.npy')
    processed_data_test = numpy.load('dataset/flower_photos_test.npy')

    train_images = processed_data_train[0]
    train_labels = processed_data_train[1]
    validation_images = processed_data_validation[0]
    validation_labels = processed_data_validation[1]
    test_images = processed_data_test[0]
    test_labels = processed_data_test[1]

    print('%d train examples, %d validation examples, %d test examples' % (len(train_images), len(validation_images), len(test_images)))

    images = tensorflow.placeholder(tensorflow.float32, [None, 299, 299, 3], 'images')
    labels = tensorflow.placeholder(tensorflow.int64, [None], 'labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, 5, True)

    trainable_variables = get_trainable_variables()
    tensorflow.losses.softmax_cross_entropy(tensorflow.one_hot(labels, 5), logits, 1)
    train_step = tensorflow.train.RMSPropOptimizer(0.0001).minimize(tensorflow.losses.get_total_loss())

    with tensorflow.name_scope('evaluation'):
        correct_prediction = tensorflow.equal(tensorflow.argmax(logits, 1), labels)
        evaluation_step = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
    
    load_fn = slim.assign_from_checkpoint_fn('dataset/pretrained/inception_v3.ckpt', get_tuned_variables(), True)

    with tensorflow.Session() as session:
        tensorflow.global_variables_initializer().run()

        print('load tuned variables from %s' % 'dataset/pretrained/inception_v3.ckpt')
        load_fn(session)

        saver = tensorflow.train.Saver()

        begin = 0
        end = 32
        for i in range(100):
            for _ in range(100):
                session.run(train_step, feed_dict={images: train_images[begin: end], labels: train_labels[begin: end]})
                begin = end
                if begin == len(train_images):
                    begin = 0
                end = begin + 32
                if end > len(train_images):
                    end = len(train_images)
            
            saver.save(session, 'dataset/model/save_model', i * 100 + 100)

            validation_accuracy = session.run(evaluation_step, feed_dict={images: validation_images, labels: validation_labels})
            print('step %d, validation accuracy %.4f%%' % (i * 100 + 100, validation_accuracy * 100))

        test_accuracy = session.run(evaluation_step, feed_dict={images: test_images, labels: test_labels})
        print('test accuracy %.4f%%' % test_accuracy * 100)

if __name__ == '__main__':
    tensorflow.app.run()