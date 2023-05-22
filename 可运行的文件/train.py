import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import numpy as np

NATURE_NUM=2.7182818284



##载入数据&数据预处理
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']#划分数据集

encoder = info.features['text'].encoder#获取预置的编码器

BUFFER_SIZE = 10000#基本参数设置
BATCH_SIZE = 64
NET_SIZE = 32

train_dataset = train_dataset.shuffle(BUFFER_SIZE)#打乱数据
train_dataset = train_dataset.padded_batch(BATCH_SIZE)#数据填充

test_dataset = test_dataset.padded_batch(BATCH_SIZE)

##创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, NET_SIZE),#嵌入层
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NET_SIZE)),#RNN层
    tf.keras.layers.Dense(NET_SIZE, activation='relu'),#全连接层1
    tf.keras.layers.Dense(1)#全连接层2
])

##设置训练参数
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

##训练模型
history = model.fit(train_dataset, epochs=8,
                    validation_data=test_dataset,
                    validation_steps=30,verbose=2)

##测试模型
results = model.evaluate(test_dataset, verbose=2)

##保存模型
#model.save('my_model.h5')



