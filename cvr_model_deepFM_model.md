```

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import (LSTM, Add, Concatenate, Dense, Embedding,
                                     Flatten, Input, Lambda, Multiply)
from tensorflow.keras.models import Model

# 示例数据，你需要用你自己的数据替换

'''
data = {
    'gender': ['M', 'F', 'M', 'F'],
    'age': [25, 30, 35, 28],
    'location': ['A', 'B', 'A', 'B'],
    'interests': ['sports', 'fashion', 'technology', 'music'],
    'device': ['mobile', 'tablet', 'mobile', 'desktop'],
    'purchase_history': [0, 1, 0, 1],
    'ad_category': ['electronics', 'fashion', 'fashion', 'electronics'],
    'advertiser': ['X', 'Y', 'Y', 'X'],
    'ad_type': ['image', 'video', 'carousel', 'image'],
    'ad_channel': ['google', 'facebook', 'twitter', 'google'],
    'ad_time': [8, 15, 21, 10],
    'page_type': ['blog', 'shopping', 'news', 'blog'],
    'page_load_speed': [2.5, 3.2, 1.8, 2.0],
    'page_layout': ['A', 'B', 'C', 'A'],
    'user_behavior': [[1, 2, 1, 3], [1, 1, 3], [1, 2, 2, 1], [3, 1, 2]],
    'CVR': [0, 1, 0, 1]
}
'''
data = {
    'gender': ['M', 'F', 'M', 'F'],
    'age': [25, 30, 35, 28],
    'user_behavior': [[1, 2, 1, 3], [1, 1, 3], [1, 2, 2, 1], [3, 1, 2]],
    'CVR': [0, 1, 0, 1]
}

df = pd.DataFrame(data)

# 数值特征
numerical_features = ['age']
numerical_data = df[numerical_features].values

# 标准化数值特征
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# 类别特征
categorical_features = ['gender']
categorical_data = df[categorical_features].values

# Count the number of unique values for each categorical feature
num_unique_values = [len(df[feature].unique())
                     for feature in categorical_features]

# Calculate the total number of unique categorical feature values
num_categorical_features = sum(num_unique_values)


# 独热编码类别特征
encoder = OneHotEncoder()
categorical_data = encoder.fit_transform(categorical_data).toarray()
print(categorical_data)

# 处理用户行为数据
max_length = max([len(x) for x in df['user_behavior']])
user_behavior_padded = tf.keras.preprocessing.sequence.pad_sequences(
    df['user_behavior'], maxlen=max_length)

# 合并所有特征
X = np.hstack([numerical_data, categorical_data])

# 特征和标签分离
y = df['CVR'].values

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# LSTM 输入处理
X_train_behavior = user_behavior_padded[:len(X_train)]
X_test_behavior = user_behavior_padded[len(X_train):]

# 其他特征输入
X_train_features = X_train_features = X_train
X_test_features = X_test

# 创建 LSTM 模型
lstm_input = Input(shape=(max_length,))
lstm_output = Embedding(max_length, 8)(lstm_input)
lstm_output = LSTM(16)(lstm_output)

# DeepFM 模型
features_input = Input(shape=(X_train_features.shape[1],))

# 线性部分
linear_output = Dense(1, activation='linear')(features_input)

# FM 部分
embedding_layer = Embedding(num_categorical_features, 8)
embeddings = []
for i in range(num_categorical_features):
    emb = embedding_layer(features_input[:, i+len(numerical_features)])
    embeddings.append(emb)

embeddings = tf.stack(embeddings, axis=1)

fm_sum_of_squares = tf.square(tf.reduce_sum(embeddings, axis=1))
fm_square_of_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
fm_output = 0.5 * (fm_sum_of_squares - fm_square_of_sum)

# DNN 部分
deep_output = Dense(64, activation='relu')(features_input)
deep_output = Dense(32, activation='relu')(deep_output)

# 合并 DeepFM 输出
# deepfm_output = Add()([linear_output, fm_output, deep_output])
deepfm_output = Concatenate()([linear_output, fm_output, deep_output])

# 合并 DeepFM 输出和 LSTM 输出
combined_output = Concatenate()([lstm_output, deepfm_output])
combined_output = Dense(32, activation='relu')(combined_output)
output = Dense(1, activation='sigmoid')(combined_output)

# 创建模型
model = Model([lstm_input, features_input], output)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy')

config = model.get_config()  # Returns pretty much every information about your model
print(config["layers"])  # returns a tuple of width, height and channels
print(len([X_train_behavior, X_train_features]))
print(X_train_behavior.shape)
print(X_train_features.shape)

# 训练模型
model.fit([X_train_behavior, X_train_features],
          y_train, epochs=3, batch_size=1, verbose=1)

# 预测
y_pred = model.predict([X_test_behavior, X_test_features])

print("预测结果：", y_pred)
```

### Example Output

```
[[0. 1.]
 [1. 0.]
 [0. 1.]
 [1. 0.]]
[{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_2'}, 'name': 'input_2', 'inbound_nodes': []}, {'class_name': 'SlicingOpLambda', 'config': {'name': 'tf.__operators__.getitem', 'trainable': True, 'dtype': 'float32', 'function': '__operators__.getitem'}, 'name': 'tf.__operators__.getitem', 'inbound_nodes': [['input_2', 0, 0, {'slice_spec': ({'start': None, 'stop': None, 'step': None}, 0)}]]}, {'class_name': 'SlicingOpLambda', 'config': {'name': 'tf.__operators__.getitem_1', 'trainable': True, 'dtype': 'float32', 'function': '__operators__.getitem'}, 'name': 'tf.__operators__.getitem_1', 'inbound_nodes': [['input_2', 0, 0, {'slice_spec': ({'start': None, 'stop': None, 'step': None}, 1)}]]}, {'class_name': 'Embedding', 'config': {'name': 'embedding_1', 'trainable': True, 'dtype': 'float32', 'batch_input_shape': (None, None), 'input_dim': 2, 'output_dim': 8, 'embeddings_initializer': {'class_name': 'RandomUniform', 'config': {'minval': -0.05, 'maxval': 0.05, 'seed': None}}, 'embeddings_regularizer': None, 'activity_regularizer': None, 'embeddings_constraint': None, 'mask_zero': False, 'input_length': None}, 'name': 'embedding_1', 'inbound_nodes': [[['tf.__operators__.getitem', 0, 0, {}]], [['tf.__operators__.getitem_1', 0, 0, {}]]]}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.stack', 'trainable': True, 'dtype': 'float32', 'function': 'stack'}, 'name': 'tf.stack', 'inbound_nodes': [[['embedding_1', 0, 0, {'axis': 1}], ['embedding_1', 1, 0, {'axis': 1}]]]}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.math.reduce_sum', 'trainable': True, 'dtype': 'float32', 'function': 'math.reduce_sum'}, 'name': 'tf.math.reduce_sum', 'inbound_nodes': [['tf.stack', 0, 0, {'axis': 1}]]}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.math.square_1', 'trainable': True, 'dtype': 'float32', 'function': 'math.square'}, 'name': 'tf.math.square_1', 'inbound_nodes': [['tf.stack', 0, 0, {'name': None}]]}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.math.square', 'trainable': True, 'dtype': 'float32', 'function': 'math.square'}, 'name': 'tf.math.square', 'inbound_nodes': [['tf.math.reduce_sum', 0, 0, {'name': None}]]}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.math.reduce_sum_1', 'trainable': True, 'dtype': 'float32', 'function': 'math.reduce_sum'}, 'name': 'tf.math.reduce_sum_1', 'inbound_nodes': [['tf.math.square_1', 0, 0, {'axis': 1}]]}, {'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, 4), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_1'}, 'name': 'input_1', 'inbound_nodes': []}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.math.subtract', 'trainable': True, 'dtype': 'float32', 'function': 'math.subtract'}, 'name': 'tf.math.subtract', 'inbound_nodes': [['tf.math.square', 0, 0, {'y': ['tf.math.reduce_sum_1', 0, 0], 'name': None}]]}, {'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 64, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_1', 'inbound_nodes': [[['input_2', 0, 0, {}]]]}, {'class_name': 'Embedding', 'config': {'name': 'embedding', 'trainable': True, 'dtype': 'float32', 'batch_input_shape': (None, None), 'input_dim': 4, 'output_dim': 8, 'embeddings_initializer': {'class_name': 'RandomUniform', 'config': {'minval': -0.05, 'maxval': 0.05, 'seed': None}}, 'embeddings_regularizer': None, 'activity_regularizer': None, 'embeddings_constraint': None, 'mask_zero': False, 'input_length': None}, 'name': 'embedding', 'inbound_nodes': [[['input_1', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense', 'inbound_nodes': [[['input_2', 0, 0, {}]]]}, {'class_name': 'TFOpLambda', 'config': {'name': 'tf.math.multiply', 'trainable': True, 'dtype': 'float32', 'function': 'math.multiply'}, 'name': 'tf.math.multiply', 'inbound_nodes': [['_CONSTANT_VALUE', -1, 0.5, {'y': ['tf.math.subtract', 0, 0], 'name': None}]]}, {'class_name': 'Dense', 'config': {'name': 'dense_2', 'trainable': True, 'dtype': 'float32', 'units': 32, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_2', 'inbound_nodes': [[['dense_1', 0, 0, {}]]]}, {'class_name': 'LSTM', 'config': {'name': 'lstm', 'trainable': True, 'dtype': 'float32', 'return_sequences': False, 'return_state': False, 'go_backwards': False, 'stateful': False, 'unroll': False, 'time_major': False, 'units': 16, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}, 'shared_object_id': 24}, 'recurrent_initializer': {'class_name': 'Orthogonal', 'config': {'gain': 1.0, 'seed': None}, 'shared_object_id': 25}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 26}, 'unit_forget_bias': True, 'kernel_regularizer': None, 'recurrent_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'recurrent_constraint': None, 'bias_constraint': None, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'implementation': 2}, 'name': 'lstm', 'inbound_nodes': [[['embedding', 0, 0, {}]]]}, {'class_name': 'Concatenate', 'config': {'name': 'concatenate', 'trainable': True, 'dtype': 'float32', 'axis': -1}, 'name': 'concatenate', 'inbound_nodes': [[['dense', 0, 0, {}], ['tf.math.multiply', 0, 0, {}], ['dense_2', 0, 0, {}]]]}, {'class_name': 'Concatenate', 'config': {'name': 'concatenate_1', 'trainable': True, 'dtype': 'float32', 'axis': -1}, 'name': 'concatenate_1', 'inbound_nodes': [[['lstm', 0, 0, {}], ['concatenate', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense_3', 'trainable': True, 'dtype': 'float32', 'units': 32, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_3', 'inbound_nodes': [[['concatenate_1', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense_4', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'sigmoid', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_4', 'inbound_nodes': [[['dense_3', 0, 0, {}]]]}]
2
(3, 4)
(3, 3)
Epoch 1/3
3/3 [==============================] - 16s 22ms/step - loss: 0.6742
Epoch 2/3
3/3 [==============================] - 0s 19ms/step - loss: 0.6386
Epoch 3/3
3/3 [==============================] - 0s 13ms/step - loss: 0.6085
1/1 [==============================] - 5s 5s/step
预测结果： [[0.52413774]]
```
