import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
d = pd.read_csv("creditcard.csv")
s = d.sample(frac=1)
e = pd.get_dummies(s, columns=['Class'])
n = e - (e-e.min()) / (e.max()-e.min())
df_x = n.drop(['Class_0','Class_1'], axis=1)
df_y = n[['Class_0', 'Class_1']]
(ar_x, ar_y) = (np.asarray(df_x, dtype='float32'), np.asarray(df_y, dtype='float32'))
split_size = int(0.8 * len(ar_x))
(train_x, train_y) = (ar_x[:split_size], ar_y[:split_size])
(test_x, test_y) = (ar_x[split_size:], ar_y[split_size:])
l, f = np.unique(d['Class'], return_counts=True)[1]
f = float(f / (l+f))
w = 1/f
train_y[:, 1] = train_y[:, 1] * w
import tensorflow as tf
i = ar_x.shape[1]
o = ar_y.shape[1]
layer_1_cells = 100
layer_2_cells = 150
tr_x = tf.placeholder(tf.float32, [None, i], name='train_x')
tr_y = tf.placeholder(tf.float32, [None, o], name='train_y')
te_x = tf.constant(test_x, name='test_x')
te_y = tf.constant(test_y, name='test_y')

/*to be continued.............*/