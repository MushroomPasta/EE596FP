'''try:
    1)change loss function
    2)shuffle x_train
    3)add audio x_train
'''


from readers import YT8MFrameFeatureReader
import tensorflow as tf
from model import Model, batch_generator
import numpy as np
a = YT8MFrameFeatureReader()
train_filenames = ['train2072.tfrecord', 'train0093.tfrecord', 'train0111.tfrecord', 'train0208.tfrecord']
data = []
with tf.Session() as sess:
#  coord = tf.train.Coordinator()
#  thread = tf.train.start_queue_runners(sess, coord)
  for train_filename in train_filenames:
     filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None)
     coord = tf.train.Coordinator()
     thread = tf.train.start_queue_runners(sess, coord)
     out = a.prepare_reader(filename_queue)
     out_np = {}
     for key, val in out.items():
#       out_np[key] = sess.run(val)
       out_np[key] = val.eval(session = sess)
     data.append(out_np)
     print()

batch_size = 2
seq_len = 300
output_size = 3862
save_path = "check"
save_every_n = 1
log_every_n = 1
max_steps = 40
generator = batch_generator(data, batch_size, seq_len)
md = Model(batch_size, seq_len, output_size)
md.train(generator, max_steps, save_path, save_every_n, log_every_n)



test_filenames = ['train2072.tfrecord', 'train0093.tfrecord', 'train0111.tfrecord', 'train0208.tfrecord']
test_data = []
with tf.Session() as sess:
  for test_filename in train_filenames:
     filename_queue = tf.train.string_input_producer([test_filename],num_epochs=None)
     coord = tf.train.Coordinator()
     thread = tf.train.start_queue_runners(sess, coord)
     out = a.prepare_reader(filename_queue)
     out_np = {}
     for key, val in out.items():
#       out_np[key] = sess.run(val)
       out_np[key] = val.eval(session = sess)
     test_data.append(out_np)

checkpoint = tf.train.latest_checkpoint(save_path)
model = Model(1, 300, output_size, training=False)
model.load(checkpoint)
generator_t = batch_generator(test_data, 1, seq_len)
lacc = []
ltp = []
lfp = []
lfn = []
count = 0
for X, y in generator_t:
  if count == 4:
      break
#  print(type(X))
  pred = model.predicts(X)
#  print(np.shape(pred))
  acc, tp, fp, fn  = model.accuracy(pred, y[299])
  lacc.append(acc)
  ltp.append(tp)
  lfp.append(fp)
  lfn.append(fn)
  print(count)
  count+=1
print(ltp,lfp,lfn,lacc)