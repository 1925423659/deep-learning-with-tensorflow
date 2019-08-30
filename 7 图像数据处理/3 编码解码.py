import tensorflow
import matplotlib.pyplot as pyplot

image_raw_data = tensorflow.gfile.FastGFile('dataset/cat.jpg', 'rb').read()

with tensorflow.Session() as session:
    image_data = tensorflow.image.decode_jpeg(image_raw_data)
    print(image_data.eval())

with tensorflow.Session() as session:
    pyplot.imshow(image_data.eval())
    pyplot.show()

with tensorflow.Session() as session:
    encode_image = tensorflow.image.encode_jpeg(image_data)
    with tensorflow.gfile.GFile('dataset/cat_output.jpg', 'wb') as f:
        f.write(encode_image.eval())