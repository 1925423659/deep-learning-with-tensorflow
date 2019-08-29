import tensorflow
import numpy
import os
import glob

extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
train_images = []
train_labels = []
validation_images = []
validation_labels = []
test_images = []
test_labels = []

def create_image_list(session, validation, test):
    sub_dir_list = [x[0] for x in os.walk('dataset/flower_photos')]
    
    for i in range(1, len(sub_dir_list)):
        sub_dir = sub_dir_list[i]
        dir_name = os.path.basename(sub_dir)
        file_list = []
        
        for extension in extensions:
            pathname = os.path.join('dataset/flower_photos', dir_name, '*.' + extension)
            file_list.extend(glob.glob(pathname))
        if not file_list:
            continue
        print('processing:', sub_dir)

        for filename in file_list:
            image_data = tensorflow.gfile.FastGFile(filename, 'rb').read()
            image = tensorflow.image.decode_jpeg(image_data)
            image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
            image = tensorflow.image.resize_images(image, [299, 299])
            image_value = session.run(image)

            chance = numpy.random.randint(100)
            if chance < validation:
                validation_images.append(image_value)
                validation_labels.append(i)
            elif chance < validation + test:
                test_images.append(image_value)
                test_labels.append(i)
            else:
                train_images.append(image_value)
                train_labels.append(i)
    
    state = numpy.random.get_state()
    numpy.random.shuffle(train_images)
    numpy.random.set_state(state)
    numpy.random.shuffle(train_labels)

    return numpy.asarray([train_images, train_labels]), numpy.asarray([validation_images, validation_labels]), numpy.asarray([test_images, test_labels])

def main():
    with tensorflow.Session() as session:
        train_data, validation_data, test_data = create_image_list(session, 10, 10)
        numpy.save('dataset/flower_photos_train.npy', train_data)
        numpy.save('dataset/flower_photos_validation.npy', validation_data)
        numpy.save('dataset/flower_photos_test.npy', test_data)

if __name__ == '__main__':
    main()