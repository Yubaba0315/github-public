
# from _future_ import 导入后续版本中的新特性，使旧版编码规范下的代码不需要大量修改就可以调用新特性
# absolute_import 从相对导入变成绝对导入,py3的import用法，加上这句话py2时候也能按照py3的import语法去正常使用
# division精确除法
# 超前使用python3的print函数
from __future__ import absolute_import, division, print_function, unicode_literals
# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras
# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

print('Tensorflow版本：', tf.__version__)

# 导入Fashion_MNIST数据集，加载数据集并返回四个NumPy数组
# train_images和train_labels数组是训练集—这是模型用来学习的数据
# 模型通过测试集（test_images与 test_labels两个数组）进行测试
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 每个图像都映射到一个标签。由于类别名称不包含在数据集中,因此把他们存储在这里以便在绘制图像时使用:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print('训练集-规模：', train_images.shape)
print('训练集-图片总数：', len(train_labels))
print('训练集-标签：', train_labels)
print('测试集-规模：', test_images.shape)
print('测试集-图片总数：', len(test_labels))
print('测试集-标签：', test_labels)

# 在训练网络之前必须对数据进行预处理。 如果您检查训练集中的第一个图像，您将看到像素值落在0到255的范围内:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()


# 在馈送到神经网络模型之前，我们将这些值缩放到0到1的范围。为此，我们将像素值值除以255（黑白）。重要的是，对训练集和测试集要以相同的方式进行预处理:
train_images = train_images / 255.0
test_images = test_images / 255.0
# 检查训练集中的第一个图像，您将看到像素值已经变为落在0到1之间
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()

# 显示训练集中的前25个图像，并在每个图像下方显示类名。验证数据格式是否正确，我们是否已准备好构建和训练网络。
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 构建模型
# 构建神经网络需要配置模型的层，然后编译模型。
# 一、设置网络层
# 一个神经网络最基本的组成部分便是网络层。网络层从提供给他们的数据中提取表示，并期望这些表示对当前的问题更加有意义
# 大多数深度学习是由串连在一起的网络层所组成。大多数网络层，例如tf.keras.layers.Dense，具有在训练期间学习的参数。
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)])

        # 网络中的第一层, tf.keras.layers.Flatten, 将图像格式从一个二维数组(包含着28x28个像素)转换成为一个包含着28 * 28 = 784个像素的一维数组。
        # 可以将这个网络层视为它将图像中未堆叠的像素排列在一起。这个网络层没有需要学习的参数;它仅仅对数据进行格式化。
        # 在像素被展平之后，网络由一个包含有两个tf.keras.layers.Dense网络层的序列组成。他们被称作“稠密链接层”或“全连接层” 。
        # 第一个Dense网络层包含有128个节点(或被称为神经元)。第二个(也是最后一个)网络层是一个包含10个节点的softmax层—它将返回包含10个概率分数的数组，总和为1。每个节点包含一个分数，表示当前图像属于10个类别之一的概率。

# 二、编译模型
# 在模型准备好进行训练之前，它还需要一些配置。这些是在模型的编译(compile)步骤中添加的:
# 损失函数 —这可以衡量模型在培训过程中的准确程度。 我们希望将此函数最小化以"驱使"模型朝正确的方向拟合。
# 优化器 —这就是模型根据它看到的数据及其损失函数进行更新的方式。
# 评价方式 —用于监控训练和测试步骤。以下示例使用准确率(accuracy)，即正确分类的图像的分数。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 三、训练模型
# 将训练数据提供给模型 - 在本案例中，他们是train_images和train_labels数组。
# 模型学习如何将图像与其标签关联
# 我们使用模型对测试集进行预测, 在本案例中为test_images数组。我们验证预测结果是否匹配test_labels数组中保存的标签。
# 通过调用model.fit方法来训练模型 — 模型对训练数据进行"拟合"。
print('---------------------START TO TRAINING---------------------')
model.fit(train_images, train_labels, epochs=5)
print('---------------------FINISHED TRAINING---------------------')
        # 随着模型训练，将显示损失和准确率等指标。该模型在训练数据上达到约0.88(或88％)的准确度。


# 四、评估准确率
# 比较模型在测试数据集上的执行情况:
print('=====================START TO EVALUATE=====================')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 五、进行预测
# 通过训练模型，我们可以使用它来预测某些图像。
predictions = model.predict(test_images)

    # 第一个预测:
print('第一个预测各结果置信度：', predictions[0])

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
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 单个对象的预测
    # 第0个图像，预测和预测数组。
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
    # 第12个图像，预测和预测数组。
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# 多个对象的预测
    # 绘制几个图像及其预测结果。正确的预测标签是蓝色的，不正确的预测标签是红色的。该数字给出了预测标签的百分比(满分100)。请注意，即使非常自信，也可能出错。
    # 绘制前X个测试图像，预测标签和真实标签以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

    # 使用训练的模型对单个图像进行预测。
    # 从测试数据集中获取图像
img = test_images[0]
print(img.shape)

    # tf.keras模型经过优化，可以一次性对批量,或者一个集合的数据进行预测。因此，即使我们使用单个图像，我们也需要将其添加到列表中:
    # 将图像添加到批次中，即使它是唯一的成员。
img = (np.expand_dims(img,0))
print(img.shape)

    # 现在来预测图像:
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

    # model.predict返回一个包含列表的列表，每个图像对应一个列表的数据。获取批次中我们(仅有的)图像的预测:
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)