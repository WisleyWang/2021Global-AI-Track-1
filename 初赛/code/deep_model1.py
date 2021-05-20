import pandas as pd
import numpy as np
import collections
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from keras import Model
from keras.layers import CuDNNLSTM,Embedding, Dense
from gensim.models import Word2Vec
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Layer
from keras.layers import Embedding, Dense, CuDNNLSTM,CuDNNGRU, Bidirectional,SpatialDropout1D,Input,\
GlobalAveragePooling1D,GlobalMaxPooling1D,Conv1D,concatenate,Dropout,Activation,BatchNormalization,Concatenate,Add,\
MaxPooling1D,Flatten,AveragePooling1D
from keras import initializers, regularizers, constraints
import keras.backend as K
'''
第三版模型，对losss进行改进，多一个二分类的loss，来看是否有病，如果有病则所有的应该都为0才对，最好的理解就是小于0，sigomod预测值尽可能
小，然后加上下一步预测的分类 ，相当于做减法，做一个修正。模型还是先用huawei的作为验证吧
'''

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


data_path='../tcdata/medical_nlp_round1_data/'
train=pd.read_csv(data_path+'track1_round1_train_20210222.csv',header=None)

test=pd.read_csv(data_path+'track1_round1_testB.csv',header=None)


### 创建训练集标签
train_labels=np.zeros((len(train),17),dtype='int8')
for cnt,i in enumerate(train[2]):
    lab=[int(x.replace('|','').strip()) for x in i.split(' ') if x and x!='|']
    for l in lab:
        train_labels[cnt,l]=1
# 合并训练集与测试集 制作特征
data=pd.concat([train,test],axis=0).reset_index(drop=True)
# 去除竖线
data[1]=data[1].apply(lambda x:x.replace('|','').strip())
train[1]=train[1].apply(lambda x:x.replace('|','').strip())
test[1]=test[1].apply(lambda x:x.replace('|','').strip())
#--------------
# ---- 这里为多loss做准备，所以制作了哥softmax的标签
encode=OneHotEncoder()
train_label1=encode.fit_transform(train_labels.sum(1).reshape(len(train),1)).toarray()
print(train_label1.shape)
cate_num=train_labels.sum(1)

# 加载w2v 预训练模型
w2v=gensim.models.KeyedVectors.load_word2vec_format(
        '../user_data/pretraining_model/w2v_128.txt', binary=False)

#分词 padding-----------------------
# 分词得到序列矩阵
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence

tokenizer = Tokenizer(lower=False, char_level=False, split=' ')
tokenizer.fit_on_texts(data[1].tolist())
seq = tokenizer.texts_to_sequences(data[1].tolist())

# 分训练和测试集合
seq = pad_sequences(seq, maxlen=100, value=0)
train_seq=np.asarray(seq[:len(train)])
test_seq=seq[len(train):]

## 得到embedding矩阵
embedding_matrix=np.zeros((len(tokenizer.word_index)+1,128))
for word in tokenizer.word_index:
    if word not in w2v.wv.vocab:
        continue
    embedding_matrix[tokenizer.word_index[word]] = w2v[word]
print(embedding_matrix.shape)

##  模型部分

def NN_huaweivv1(maxlen, embedding_matrix=None, class_num=17):
	#
	emb_layer = Embedding(
		embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix],
		trainable=False,
	)
	seq1 = Input(shape=(maxlen,))
	
	x1 = emb_layer(seq1)
	sdrop = SpatialDropout1D(rate=0.2)
	lstm_layer = Bidirectional(CuDNNGRU(128, return_sequences=True))
	gru_layer = Bidirectional(CuDNNGRU(128, return_sequences=True))
	cnn1d_layer = Conv1D(64, kernel_size=12, padding="same", kernel_initializer="he_uniform")
	x1 = TimestepDropout(0.2)(x1)
	x1 = sdrop(x1)
	lstm1 = lstm_layer(x1)
	gru1 = gru_layer(lstm1)
	att_1 = Attention(maxlen)(lstm1)
	att_2 = Attention(maxlen)(gru1)
	cnn1 = cnn1d_layer(lstm1)
	
	avg_pool = GlobalAveragePooling1D()
	max_pool = GlobalMaxPooling1D()
	
	x1 = concatenate([att_1, att_2, Attention(maxlen)(cnn1), avg_pool(cnn1), max_pool(cnn1)])
	
	x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(256)(x1))))
	x = Activation(activation="relu")(BatchNormalization()(Dense(128)(x)))
	pred = Dense(class_num, activation='sigmoid')(x)
	model = Model(inputs=seq1, outputs=pred)
	return model


def NN_huaweiv1(maxlen, embedding_matrix=None, class_num=17):
	emb_layer = Embedding(
		embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=maxlen, weights=[embedding_matrix],
		trainable=False,
	)
	seq1 = Input(shape=(maxlen,), name='seq1')
	
	x1 = emb_layer(seq1)
	sdrop = SpatialDropout1D(rate=0.2)
	lstm_layer = Bidirectional(CuDNNLSTM(128, return_sequences=True))
	gru_layer = Bidirectional(CuDNNGRU(128, return_sequences=True))
	cnn1d_layer = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="he_uniform")
	#     x1=TimestepDropout(0.2)(x1)
	x1 = sdrop(x1)
	lstm1 = lstm_layer(x1)
	gru1 = gru_layer(lstm1)
	att_1 = Attention(maxlen)(lstm1)
	att_2 = Attention(maxlen)(gru1)
	cnn1 = cnn1d_layer(lstm1)
	
	avg_pool = GlobalAveragePooling1D()
	max_pool = GlobalMaxPooling1D()
	
	x1 = concatenate([att_1, att_2, avg_pool(cnn1), max_pool(cnn1), avg_pool(gru1), max_pool(gru1)])
	#     hin = Input(shape=(num_feature_input, ))
	#     htime = Dense(16, activation='relu')(hin)
	
	#     x = concatenate([x1, x2, merge, htime])
	
	x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(128)(x1))))
	x = Activation(activation="relu")(BatchNormalization()(Dense(64)(x)))
	pred_den1 = Dense(9)(x)
	pred1 = Activation(activation='softmax', name='pred1')(pred_den1)
	
	xx = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(128)(x))))
	out2 = concatenate([xx, pred_den1])
	out2 = Dense(class_num, activation='sigmoid', name='out2')(out2)
	model = Model(inputs=seq1, outputs=[pred1, out2])
	#     from keras.utils import multi_gpu_model
	#     model = multi_gpu_model(model, 2)
	return model


class TimestepDropout(Dropout):
	def __init__(self, rate, **kwargs):
		super(TimestepDropout, self).__init__(rate, **kwargs)
		self.input_spec = InputSpec(ndim=3)
	
	def _get_noise_shape(self, inputs):
		input_shape = K.shape(inputs)
		noise_shape = (input_shape[0], 1, input_shape[2])
		return noise_shape

##
import tensorflow as tf
from keras.layers import InputSpec
class Attention(Layer):
	def __init__(self, step_dim,
	             W_regularizer=None, b_regularizer=None,
	             W_constraint=None, b_constraint=None,
	             bias=True, **kwargs):
		"""
		Keras Layer that implements an Attention mechanism for temporal data.
		Supports Masking.
		Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
		# Input shape
			3D tensor with shape: `(samples, steps, features)`.
		# Output shape
			2D tensor with shape: `(samples, features)`.
		:param kwargs:
		Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
		The dimensions are inferred based on the output shape of the RNN.
		Example:
			# 1
			model.add(LSTM(64, return_sequences=True))
			model.add(Attention())
			# next add a Dense layer (for classification/regression) or whatever...
			# 2
			hidden = LSTM(64, return_sequences=True)(words)
			sentence = Attention()(hidden)
			# next add a Dense layer (for classification/regression) or whatever...
		"""
		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')
		
		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		
		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)
		
		self.bias = bias
		self.step_dim = step_dim
		self.features_dim = 0
		
		super(Attention, self).__init__(**kwargs)
	
	def build(self, input_shape):
		assert len(input_shape) == 3
		#         print('-------------',type(input_shape))
		self.W = self.add_weight(name='{}_W'.format(self.name),
		                         shape=(input_shape[-1],),
		                         initializer=self.init,
		                         regularizer=self.W_regularizer,
		                         constraint=self.W_constraint)
		self.features_dim = input_shape[-1]
		
		if self.bias:
			self.b = self.add_weight(name='{}_b'.format(self.name),
			                         shape=(input_shape[1],),
			                         initializer='zero',
			                         regularizer=self.b_regularizer,
			                         constraint=self.b_constraint)
		else:
			self.b = None
		
		self.built = True
	
	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None
	
	def call(self, x, mask=None):
		features_dim = self.features_dim
		step_dim = self.step_dim
		
		e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
		              (-1, step_dim))  # e = K.dot(x, self.W)
		if self.bias:
			e += self.b
		e = K.tanh(e)
		
		a = K.exp(e)
		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			# cast the mask to floatX to avoid float64 upcasting in theano
			a *= K.cast(mask, K.floatx())
		# in some cases especially in the early stages of training the sum may be almost zero
		# and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)
		
		c = K.sum(a * x, axis=1)
		return c
	
	def compute_output_shape(self, input_shape):
		return input_shape[0], self.features_dim


class Lookahead(object):
	"""Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
	"""
	
	def __init__(self, k=5, alpha=0.5):
		self.k = k
		self.alpha = alpha
		self.count = 0
	
	def inject(self, model):
		"""Inject the Lookahead algorithm for the given model.
		The following code is modified from keras's _make_train_function method.
		See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
		"""
		if not hasattr(model, 'train_function'):
			raise RuntimeError('You must compile your model before using it.')
		
		model._check_trainable_weights_consistency()
		
		if model.train_function is None:
			inputs = (model._feed_inputs +
			          model._feed_targets +
			          model._feed_sample_weights)
			if model._uses_dynamic_learning_phase():
				inputs += [K.learning_phase()]
			fast_params = model._collected_trainable_weights
			
			with K.name_scope('training'):
				with K.name_scope(model.optimizer.__class__.__name__):
					training_updates = model.optimizer.get_updates(
						params=fast_params,
						loss=model.total_loss)
					slow_params = [K.variable(p) for p in fast_params]
				fast_updates = (model.updates +
				                training_updates +
				                model.metrics_updates)
				
				slow_updates, copy_updates = [], []
				for p, q in zip(fast_params, slow_params):
					slow_updates.append(K.update(q, q + self.alpha * (p - q)))
					copy_updates.append(K.update(p, q))
				
				# Gets loss and metrics. Updates weights at each call.
				fast_train_function = K.function(
					inputs,
					[model.total_loss] + model.metrics_tensors,
					updates=fast_updates,
					name='fast_train_function',
					**model._function_kwargs)
				
				def F(inputs):
					self.count += 1
					R = fast_train_function(inputs)
					if self.count % self.k == 0:
						K.batch_get_value(slow_updates)
						K.batch_get_value(copy_updates)
					return R
				
				model.train_function = F


class TimestepDropout(Dropout):
	def __init__(self, rate, **kwargs):
		super(TimestepDropout, self).__init__(rate, **kwargs)
		self.input_spec = InputSpec(ndim=3)
	
	def _get_noise_shape(self, inputs):
		input_shape = K.shape(inputs)
		noise_shape = (input_shape[0], 1, input_shape[2])
		return noise_shape


def multilabel_categorical_crossentropy(y_true, y_pred):
	"""多标签分类的交叉熵
	说明：y_true和y_pred的shape一致，y_true的元素非0即1，
		 1表示对应的类为目标类，0表示对应的类为非目标类。
	"""
	y_pred = (1 - 2 * y_true) * y_pred
	y_pred_neg = y_pred - y_true * 1e12
	y_pred_pos = y_pred - (1 - y_true) * 1e12
	zeros = K.zeros_like(y_pred[..., :1])
	y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
	y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
	neg_loss = K.logsumexp(y_pred_neg, axis=-1)
	pos_loss = K.logsumexp(y_pred_pos, axis=-1)
	return neg_loss + pos_loss


def multi_category_focal_loss2(gamma=2., alpha=.25):
	"""
	focal loss for multi category of multi label problem
	适用于多分类或多标签问题的focal loss
	alpha控制真值y_true为1/0时的权重
		1的权重为alpha, 0的权重为1-alpha
	当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
	当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
	当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
		尝试将alpha调大,鼓励模型进行预测出1。
	Usage:
	 model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
	"""
	epsilon = 1.e-7
	gamma = float(gamma)
	alpha = tf.constant(alpha, dtype=tf.float32)
	
	def multi_category_focal_loss2_fixed(y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
		
		alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
		y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
		ce = -tf.log(y_t)
		weight = tf.pow(tf.subtract(1., y_t), gamma)
		fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
		loss = tf.reduce_mean(fl)
		return loss
	
	return multi_category_focal_loss2_fixed


def multi_category_focal_loss2_fixed(y_true, y_pred):
	epsilon = 1.e-7
	gamma = 2.
	alpha = tf.constant(0.25, dtype=tf.float32)
	
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
	
	alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
	y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
	ce = -tf.log(y_t)
	weight = tf.pow(tf.subtract(1., y_t), gamma)
	fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
	loss = tf.reduce_mean(fl) * 100
	return loss
# 评价指标*10
def mean_pred(y_true, y_pred):
	return -K.mean(y_true * K.log(y_pred + 1.e-7) + (1 - y_true) * K.log(1 - y_pred + 1.e-7)) * 10

## training---------------------
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from keras.utils import multi_gpu_model

batch_size = 128
epochs = 100
weight_name = 'deep_model1_vv1'
oof = np.zeros((len(train), 17))
tmp = 0
test_oof = np.zeros((len(test), 17))

folds = StratifiedKFold(n_splits=8, shuffle=True, random_state=2018)  # 2018
for fold_n, (trn_idx, val_idx) in enumerate(folds.split(train, cate_num)):
	print('Fold', fold_n)
	print('Build model...')
	print('正样本比例:', train_labels[trn_idx].mean(0))
	model = NN_huaweivv1(maxlen=100, embedding_matrix=embedding_matrix)
	
	# model = multi_gpu_model(model, gpus=2)# 多GPU训练
	model.compile('adam', ['binary_crossentropy'], metrics=['accuracy', mean_pred])
	lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
	lookahead.inject(model)  # add into model
	print('Train...')
	early_stopping = EarlyStopping(monitor='val_mean_pred', patience=2, mode='min')
	check_point = ModelCheckpoint('../user_data/model_data/deep_model1/%s_%d.h5' % (weight_name, fold_n), monitor='val_mean_pred', verbose=1,
	                              save_best_only=True, save_weights_only=True)
	
	model.fit(train_seq[trn_idx], train_labels[trn_idx],
	          batch_size=batch_size,
	          epochs=epochs,
	          callbacks=[early_stopping, check_point],
	          validation_data=(train_seq[val_idx], train_labels[val_idx]))

	model.load_weights('../user_data/model_data/deep_model1/%s_%d.h5' % (weight_name, fold_n))
	oof[val_idx, :] = model.predict(train_seq[val_idx], batch_size=batch_size)
	tmp_test_pred = model.predict(test_seq, batch_size=batch_size)
	test_oof[:, :] += tmp_test_pred / folds.n_splits

# 保存模型---
sub=pd.DataFrame()
sub['report_ID']=test[0]
sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in test_oof ]
sub.to_csv('../prediction_result/B_vv1.csv',index=False,header=False)

# training2---------------------------------------------------------

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from keras.utils import multi_gpu_model

batch_size = 128
epochs = 100
weight_name = 'deep_model1_v1'
oof = np.zeros((len(train), 17))
tmp = 0
test_oof = np.zeros((len(test), 17))

folds = StratifiedKFold(n_splits=8, shuffle=True, random_state=2018)  # 2018
for fold_n, (trn_idx, val_idx) in enumerate(folds.split(train, cate_num)):
	print('Fold', fold_n)
	print('Build model...')
	print('正样本比例:', train_labels[trn_idx].mean(0))

	model = NN_huaweiv1(maxlen=100, embedding_matrix=embedding_matrix)

	# model = multi_gpu_model(model, gpus=2) # 多gpu训练
	model.compile('adam', ['categorical_crossentropy','binary_crossentropy'], metrics=['accuracy', mean_pred])
	lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
	lookahead.inject(model)  # add into model
	print('Train...')
	early_stopping = EarlyStopping(monitor='val_out2_mean_pred', patience=2, mode='min')
	check_point = ModelCheckpoint('../user_data/model_data/deep_model1/%s_%d.h5' % (weight_name, fold_n), monitor='val_out2_mean_pred', verbose=1,
	                              save_best_only=True, save_weights_only=True)
	trn_label = {'pred1': train_label1[trn_idx], 'out2': train_labels[trn_idx]}
	val_label = {'pred1': train_label1[val_idx], 'out2': train_labels[val_idx]}
	model.fit(train_seq[trn_idx], trn_label,
	          batch_size=batch_size,
	          epochs=epochs,
	          callbacks=[early_stopping, check_point],
	          validation_data=(train_seq[val_idx], val_label))
	
	model.load_weights('../user_data/model_data/deep_model1/%s_%d.h5' % (weight_name, fold_n))
	_, oof[val_idx, :] = model.predict(train_seq[val_idx], batch_size=batch_size)
	_, tmp_test_pred = model.predict(test_seq, batch_size=batch_size)
	test_oof[:, :] += tmp_test_pred / folds.n_splits

sub=pd.DataFrame()
sub['report_ID']=test[0]
sub['Prediction']=[ '|'+' '.join(['%.12f'%j for j in i]) for i in test_oof ]
sub.to_csv('../prediction_result/B_v1.csv',index=False,header=False)