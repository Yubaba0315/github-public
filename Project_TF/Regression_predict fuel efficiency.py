from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
# import pathlib
import matplotlib.pyplot as plt
import pandas as pd
# 使用seaborn进行pairplot数据可视化，安装命令
import seaborn as sns
print('Tensorflow版本：', tf.__version__)

# 1.1 下载数据集
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print('数据存放路径：', dataset_path)
# 用pandas导入数据
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

# 1.2 数据清理,删除未知数据
print('数据集中的未知内容：', dataset.isna().sum())
dataset = dataset.dropna()

# “Origin”这一列实际上是分类(国家)，而不是数字。 所以把它转换为独热编码：
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

# 1.3. 将数据分为训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 1.4   检查数据
# 1.4.1 快速浏览训练集中几对列的联合分布：
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()
# 1.4.2 查看整体统计数据：
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# 1.5. 从标签中分割特征
# 将目标值或“标签”与特征分开，此标签是您训练的模型进行预测的值：
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 1.6. 标准化数据
# 再看一下上面的train_stats块，注意每个特征的范围有多么不同。
# 使用不同的比例和范围对特征进行标准化是一个很好的实践，虽然模型可能在没有特征标准化的情况下收敛，但它使训练更加困难，并且它使得最终模型取决于输入中使用的单位的选择。
# 注意：尽管我们仅从训练数据集中有意生成这些统计信息，但这些统计信息也将用于标准化测试数据集。我们需要这样做，将测试数据集投影到模型已经训练过的相同分布中。
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 2.1. 构建模型
# 让我们建立我们的模型。在这里，我们将使用具有两个密集连接隐藏层的Sequential模型，以及返回单个连续值的输出层。
# 模型构建步骤包含在函数build_model中，因为我们稍后将创建第二个模型。
def build_model():
  model = keras.Sequential([
          keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
          keras.layers.Dense(64, activation=tf.nn.relu),
          keras.layers.Dense(1)])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
model = build_model()

# 2.2. 检查模型
# 使用.summary方法打印模型的简单描述
model.summary()

# 现在试试这个模型。从训练数据中取出一批10个样本数据并在调用model.predict函数。
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# 2.3. 训练模型
# 训练模型1000个周期，并在history对象中记录训练和验证准确性：
# 通过为每个完成的周期打印“>”来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('>', end='')
EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# 使用存储在history对象中的统计数据可视化模型的训练进度。
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()
plot_history(history)
# 该图表显示在约100个周期之后，验证误差几乎没有改进，甚至降低。
# 让我们更新model.fit调用，以便在验证分数没有提高时自动停止训练。我们将使用EarlyStopping回调来测试每个周期的训练状态。
# 如果经过一定数量的周期而没有显示出改进，则自动停止训练。

model = build_model()
# “patience”参数是检查改进的周期量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)
# 上图显示在验证集上平均误差通常约为+/-2MPG

# 使用测试集来看一下泛化模型效果，我们在训练模型时没有使用测试集
# 当我们在现实世界中使用模型时，我们可以期待模型预测。
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# 2.4. 预测
# 最后，使用测试集中的数据预测MPG值：
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# 看起来我们的模型预测得相当好，我们来看看错误分布：
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

# 3. 结论
# 本章节介绍了一些处理回归问题的技巧：
# 均方误差（MSE）是用于回归问题的常见损失函数（不同的损失函数用于分类问题）。
# 同样，用于回归的评估指标与分类不同，常见的回归度量是平均绝对误差（MAE）。
# 当数字输入数据特征具有不同范围的值时，应将每个特征独立地缩放到相同范围。
# 如果没有太多训练数据，应选择隐藏层很少的小网络，以避免过拟合。
# 尽早停止是防止过拟合的有效技巧。