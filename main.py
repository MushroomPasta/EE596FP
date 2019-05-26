from readers import YT8MFrameFeatureReader
import tensorflow as tf
from model import Model, batch_generator
a = YT8MFrameFeatureReader()
train_filenames = ['train2072.tfrecord', 'train0093.tfrecord', 'train0111.tfrecord', 'train0208.tfrecord']
data = []
for train_filename in train_filenames:
   filename_queue = tf.train.string_input_producer([train_filename],num_epochs=None)
   out = a.prepare_reader(filename_queue)
   data.append(out)

batch_size = 2
seq_len = 300
output_size = 3862
save_path = "check"
save_every_n = 1
log_every_n = 1
max_steps = 20
generator = batch_generator(data, batch_size, seq_len)
md = Model(batch_size, seq_len, output_size)
md.train(generator, max_steps, save_path, save_every_n, log_every_n)