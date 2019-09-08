import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
import os
import pandas as pd
pd.set_option('display.width', None)    # 设置pandas列表打印时，宽度方向能完全显示
pd.set_option('display.max_row', None)  # 设置pandas列表打印时，条数方向能完全显示
import numpy as np
np.set_printoptions(threshold=np.inf)
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
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# class PrintDot(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs):
#         if epoch % 2 == 0: print('完成训练周期epoch：',epoch)
#         print('>', end='')

print('---------------------训 练 开 始---------------------')
model.fit(train_images, train_labels,
          batch_size=100,
          epochs=50,
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


# 五、进行预测
# 通过训练模型，我们可以使用它来预测某些图像。
predictions = model.predict(test_images)
print('第一个预测各结果置信度：'+'\n', predictions[0])
# 预测是10个数字的数组。这些描述了模型的"信心"，即图像对应于10种不同服装中的每一种。我们可以看到哪个标签具有最高的置信度值：
# 哪个标签具有最高的置信度值
print('预测结果中具有最高的置信度值的标签：', np.argmax(predictions[0]))
print('图片对应的真实标签：', test_labels[0])

# 用图表来查看全部10个类别
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                100 * np.max(predictions_array),
                class_names[true_label]),
                color=color)
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xlabel('Predict Label')
    plt.ylabel('Probability')
    plt.xticks(range(10), class_names, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# # 单个对象的预测
#     # 第0个图像，预测和预测数组。
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, org_test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()
#     # 第12个图像，预测和预测数组。
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, org_test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

# 多个对象的预测
#     绘制几个图像及其预测结果。正确的预测标签是蓝色的，不正确的预测标签是红色的。该数字给出了预测标签的百分比(满分100)。请注意，即使非常自信，也可能出错。
#     绘制前X个测试图像，预测标签和真实标签以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 10
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*4*num_cols, 4*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, org_test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# # 使用训练的模型对单个图像进行预测。
# # 从测试数据集中获取图像
# img = test_images[0]
# print(img.shape)
#     # tf.keras模型经过优化，可以一次性对批量,或者一个集合的数据进行预测。因此，即使我们使用单个图像，我们也需要将其添加到列表中:
#     # 将图像添加到批次中，即使它是唯一的成员。
# img = (np.expand_dims(img,0))
# print(img.shape)
#     # 现在来预测图像:
# predictions_single = model.predict(img)
# print(predictions_single)
# plot_value_array(0, predictions_single, test_labels)
# plt.xticks(range(10), class_names, rotation=45)
# plt.show()
#
#     # model.predict返回一个包含列表的列表，每个图像对应一个列表的数据。获取批次中我们(仅有的)图像的预测:
# prediction_result = np.argmax(predictions_single[0])
# print(prediction_result)