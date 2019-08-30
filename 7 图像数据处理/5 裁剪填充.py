import tensorflow
import matplotlib.pyplot as pyplot

image_raw = tensorflow.gfile.FastGFile('dataset/cat.jpg', 'rb').read()
image = tensorflow.image.decode_jpeg(image_raw)

with tensorflow.Session() as session:
    crop_image = tensorflow.image.resize_image_with_crop_or_pad(image, 1000, 1000)

    pyplot.imshow(crop_image.eval())
    pyplot.show()

with tensorflow.Session() as session:
    pad_image = tensorflow.image.resize_image_with_crop_or_pad(image, 3000, 3000)

    pyplot.imshow(pad_image.eval())
    pyplot.show()