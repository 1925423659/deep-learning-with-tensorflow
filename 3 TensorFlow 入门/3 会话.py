import tensorflow

a = tensorflow.constant([1.0, 2.0], name='a')
b = tensorflow.constant([2.0, 3.0], name='b')
result = a + b

session = tensorflow.Session()
print(session.run(result))
session.close()

with tensorflow.Session() as session:
    print(session.run(result))

session = tensorflow.Session()
with session.as_default():
    print(result.eval())

session = tensorflow.Session()
print(session.run(result))
print(result.eval(session=session))

session = tensorflow.InteractiveSession()
print(result.eval())
session.close()

config = tensorflow.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
session_1 = tensorflow.InteractiveSession(config=config)
session_2 = tensorflow.Session(config=config)