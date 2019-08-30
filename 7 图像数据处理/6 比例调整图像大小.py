import tensorflow
import matplotlib.pyplot as pyplot

image_raw = tensorflow.gfile.FastGFile('dataset/cat.jpg', 'rb').read()
image = tensorflow.image.decode_jpeg(image_raw)

with tensorflow.Session() as session:
    print(image.eval().shape)

    crop_image = tensorflow.image.central_crop(image, 0.5)

    pyplot.imshow(crop_image.eval())
    pyplot.show()