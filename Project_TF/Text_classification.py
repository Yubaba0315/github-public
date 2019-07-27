import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
print('Tensorflow版本：', tf.__version__)

# TensorFlow 中包含 IMDB 数据集。我们已对该数据集进行了预处理，将影评（字词序列）转换为整数序列，其中每个整数表示字典中的一个特定字词。
# 以下代码会将 IMDB 数据集下载到您的计算机上（如果您已下载该数据集，则会使用缓存副本）：
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print('训练集规模：',len(train_data))
print('测试集规模：',len(test_data))
    # 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。

# 探索数据
# 我们花点时间来了解一下数据的格式。该数据集已经过预处理：每个样本都是一个整数数组，表示影评中的字词。每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# 影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词。第一条影评如下所示：
print(train_data[0])

# 影评的长度可能会有所不同。以下代码显示了第一条和第二条影评中的字词数。由于神经网络的输入必须具有相同长度，因此我们稍后需要解决此问题。
len(train_data[0]), len(train_data[1])

# 将整数转换回字词
# 了解如何将整数转换回文本可能很有用。在以下代码中，我们将创建一个辅助函数来查询包含整数到字符串映射的字典对象：
    # A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
    # The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print('以下为由整数转回的字词：', decode_review(train_data[0]))

# 准备数据
# 影评（整数数组）必须转换为张量，然后才能馈送到神经网络中。我们可以通过以下两种方法实现这种转换：
# 对数组进行独热编码，将它们转换为由 0 和 1 构成的向量。例如，序列 [3, 5] 将变成一个 10000 维的向量，除索引 3 和 5 转换为 1 之外，其余全转换为 0。然后，将它作为网络的第一层，一个可以处理浮点向量数据的密集层。不过，这种方法会占用大量内存，需要一个大小为 num_words * num_reviews 的矩阵。
# 或者，我们可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层。
# 在本教程中，我们将使用第二种方法。
# 由于影评的长度必须相同，我们将使用 pad_sequences 函数将长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 现在，我们来看看样本的长度：
len(train_data[0]), len(train_data[1])
# 并检查（现已填充的）第一条影评：
print(train_data[0])


# 构建模型
# 神经网络通过堆叠层创建而成，这需要做出两个架构方面的主要决策：
        # 要在模型中使用多少个层？
        # 要针对每个层使用多少个隐藏单元？
# 在本示例中，输入数据由字词-索引数组构成。要预测的标签是 0 或 1。接下来，我们为此问题构建一个模型：
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
        # 第一层是 Embedding 层。该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。模型在接受训练时会学习这些向量。这些向量会向输出数组添加一个维度。生成的维度为：(batch, sequence, embedding)。
        # 接下来，一个 GlobalAveragePooling1D 层通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。这样，模型便能够以尽可能简单的方式处理各种长度的输入。
        # 该长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）。
        # 最后一层与单个输出节点密集连接。应用 sigmoid 激活函数后，结果是介于 0 到 1 之间的浮点值，表示概率或置信水平。
# 损失函数和优化器
# 模型在训练时需要一个损失函数和一个优化器。由于这是一个二元分类问题且模型会输出一个概率（应用 S 型激活函数的单个单元层），因此我们将使用 binary_crossentropy 损失函数。
# 该函数并不是唯一的损失函数，例如，您可以选择 mean_squared_error。但一般来说，binary_crossentropy 更适合处理概率问题，它可测量概率分布之间的“差距”，在本例中则为实际分布和预测之间的“差距”。
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集
# 目标是仅使用训练数据开发和调整模型，然后仅使用一次测试数据评估准确率。在训练时，我们需要检查模型处理从未见过的数据的准确率
# 我们从原始训练数据中分离出 10000 个样本，创建一个验证集。
x_val = train_data[:10000]
y_val = train_labels[:10000]

# 第10000个以后的数据作为训练集
partial_x_train = train_data[10000:]
partial_y_train = train_labels[10000:]

# 训练模型
# 用有 512 个样本的小批次训练模型 40 个周期。这将对 x_train 和 y_train 张量中的所有样本进行 40 次迭代。在训练期间，监控模型在验证集的 10000 个样本上的损失和准确率：
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型
# 我们来看看模型的表现如何。模型会返回两个值：损失（表示误差的数字，越低越好）和准确率。
results = model.evaluate(test_data, test_labels)
print(results)

# 创建准确率和损失随时间变化的图
# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包括训练期间发生的所有情况：

history_dict = history.history
print(history_dict.keys())


# 一共有 4 个条目：每个条目对应训练和验证期间的一个受监控指标。我们可以使用这些指标绘制训练损失与验证损失图表以进行对比，并绘制训练准确率与验证准确率图表：

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 在该图表中，圆点表示训练损失和准确率，实线表示验证损失和准确率。
# 可以注意到，训练损失随着周期数的增加而降低，训练准确率随着周期数的增加而提高。在使用梯度下降法优化模型时，这属于正常现象 - 该方法应在每次迭代时尽可能降低目标值。
# 验证损失和准确率的变化情况并非如此，它们似乎在大约 20 个周期后达到峰值。这是一种过拟合现象：模型在训练数据上的表现要优于在从未见过的数据上的表现。在此之后，模型会过度优化和学习特定于训练数据的表示法，而无法泛化到测试数据。