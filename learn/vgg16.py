# 直接导入已经训练好的VGG16网络
from keras.applications.vgg16 import VGG16
# load_image作用是载入图片
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

model = VGG16()
# 参数target_size用于设置目标的大小，如此一来无论载入的原图像大小如何，都会被标准化成统一的大小，这样做是为了向神经网络中方便地输入数据所需的。
image = load_img('elephant.jpg',
                 target_size=(224, 224))
# 函数img_to_array会把图像中的像素数据转化成NumPy中的array，这样数据才可以被Keras所使用。
image = img_to_array(image)
# 神经网络接收一张或多张图像作为输入，也就是说，输入的array需要有4个维度： samples, rows, columns, and channels。由于我们仅有一个 sample（即一张image），我们需要对这个array进行reshape操作。
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# 对图像进行预处理
image = preprocess_input(image)
# 预测图像的类别
y = model.predict(image)
# Keras提供了一个函数decode_predictions()，用以对已经得到的预测向量进行解读。该函数返回一个类别列表，以及类别中每个类别的预测概率，
label = decode_predictions(y)
label = label[0][0]
print('%s(%.2f%%)' % (label[1], label[2] * 100))
