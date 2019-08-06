import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
import os
import pandas as pd
pd.set_option('display.width', None)    # 设置pandas列表打印时，宽度方向能完全显示
pd.set_option('display.max_row', None)  # 设置pandas列表打印时，条数方向能完全显示
# import numpy as np
# np.set_printoptions(threshold=np.inf)
# 使用seaborn进行pairplot数据可视化
import seaborn as sns

import numpy as np
np.set_printoptions(threshold=np.inf)           # 设置numpy列表打印时，宽度方向能完全显示
import matplotlib.pyplot as plt
print('Tensorflow版本：', tf.__version__)

# 导入Fashion_MNIST数据集，加载数据集并返回四个NumPy数组
# train_images和train_labels数组是训练集—这是模型用来学习的数据
# 模型通过测试集（test_images与 test_labels两个数组）进行测试
fashion_mnist = keras.datasets.fashion_mnist
(org_train_images, train_labels), (org_test_images, test_labels) = fashion_mnist.load_data()
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print('原始训练集规格：', org_train_images.shape)
print('原始测试集规格：', org_test_images.shape)
train_images = org_train_images.reshape(-1, 28, 28, 1)
test_images = org_test_images.reshape(-1, 28, 28, 1)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 每个图像都映射到一个标签。由于类别名称不包含在数据集中,因此把他们存储在这里以便在绘制图像时使用:
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 在馈送到神经网络模型之前，我们将这些值缩放到0到1的范围。为此，我们将像素值值除以255（黑白）。重要的是，对训练集和测试集要以相同的方式进行预处理:
train_images = train_images / 255.0
test_images = test_images / 255.0
print('训练集规格：', train_images.shape)
print('训练集-图片总数：', len(train_labels))
print('测试集规格：', test_images.shape)
print('测试集-图片总数：', len(test_labels))


model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('神经网络基本信息')
model.summary()
print('\n','---------------------评估未训练模型---------------------')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('初始模型预测准确率：{:5.2f}%'.format(100*test_acc))

tbCallBack = keras.callbacks.TensorBoard(log_dir='./FashonMNIST_conv2D',
                                         histogram_freq=1,
                                         write_graph=True,
                                         write_grads=True,
                                         write_images=True)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# class PrintDot(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs):
#         if epoch % 2 == 0: print('完成训练周期epoch：',epoch)
#         print('>', end='')

print('---------------------训 练 开 始---------------------')
model.fit(train_images, train_labels,
          batch_size=100,
          epochs=1,
          callbacks=[
                     # PrintDot(),
                     early_stop,
                     # cp_callback,
                     tbCallBack],
          validation_split=0.2,
          validation_data=(test_images, test_labels),
          verbose=1)
# 调用tensorboard：工作目录打开cmd，输入  tensorboard --logdir=FashonMNIST_conv2D --host=127.0.0.1
print('---------------------训 练 结 束---------------------')

print('-----------------训 练 后 模 型 评 估----------------')
test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=1)
print('模型预测准确率：{:5.2f}%'.format(100*test_acc))
