import tensorflow
import matplotlib.pyplot as pyplot

image_raw = tensorflow.gfile.FastGFile('dataset/cat.jpg', 'rb').read()
image = tensorflow.image.decode_jpeg(image_raw)

with tensorflow.Session() as session:
    image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
    print(image.eval().shape)
    resize_image = tensorflow.image.resize_images(image, [300, 300], 0)

    pyplot.imshow(resize_image.eval())
    pyplot.show()

with tensorflow.Session() as session:
    image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
    print(image.eval().shape)
    resize_image = tensorflow.image.resize_images(image, [300, 300], 1)

    pyplot.imshow(resize_image.eval())
    pyplot.show()

with tensorflow.Session() as session:
    image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
    print(image.eval().shape)
    resize_image = tensorflow.image.resize_images(image, [300, 300], 2)

    pyplot.imshow(resize_image.eval())
    pyplot.show()

with tensorflow.Session() as session:
    image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
    print(image.eval().shape)
    resize_image = tensorflow.image.resize_images(image, [300, 300], 3)

    pyplot.imshow(resize_image.eval())
    pyplot.show()