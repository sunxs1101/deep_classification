# -.- coding: utf-8 -.-
import tensorflow as tf
import numpy as np

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("num_gpus", 1, "number of gpus (default: 1)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#decode config
tf.flags.DEFINE_string("model_path", "./runs/checkpoints_server/best_checkpoints", "model path ")
#tf.flags.DEFINE_string("data_path", "./data/data_server", "数据存放文件夹")
tf.flags.DEFINE_string("label_inv_path", "./runs/label_inv_server", "dir where label_inv are")

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9

class TextCNN(object):
  """
  A CNN for text classification.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  """
  def __init__(
    self, sequence_length, num_classes, vocab_size,
    embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

      # Placeholders for input, output and dropout
      self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
      self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0)
      # Embedding layer
      with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        W = _variable_on_cpu('W',[vocab_size,embedding_size],
            tf.random_uniform_initializer( -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

      # Create a convolution + maxpool layer for each filter size
      pooled_outputs = []
      for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
          filter_shape = [filter_size, embedding_size, 1, num_filters]
          W = _variable_on_cpu('W', filter_shape, tf.truncated_normal_initializer( stddev=0.1))
          b = _variable_on_cpu('b', num_filters,tf.constant_initializer(0.1))
          conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
          # Maxpooling over the outputs
          pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
          pooled_outputs.append(pooled)

      # Combine all the pooled features
      num_filters_total = num_filters * len(filter_sizes)
      self.h_pool = tf.concat(pooled_outputs, 3)
      self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

      # Add dropout
      with tf.variable_scope("dropout"):
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

      # Final (unnormalized) scores and predictions
      with tf.variable_scope("output"):
        W = _variable_on_cpu('W', [num_filters_total, num_classes], tf.truncated_normal_initializer(stddev=0.1))
        b = _variable_on_cpu('b', [num_classes], tf.constant_initializer(0.1))
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
        self.probs= tf.nn.softmax(self.scores,name="probs")
      # CalculateMean cross-entropy loss
      with tf.variable_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        compose_losses = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        tf.add_to_collection("losses",compose_losses)
        self.loss = compose_losses

      # Accuracy
      with tf.variable_scope("accuracy"):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

class NetConfig():
  def __init__(self, sequence_length, num_classes, vocab_size,
    embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, dropout_keep_prob = 0.5):
    self.sequence_length = sequence_length
    self.num_classes = num_classes
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.filter_sizes = filter_sizes
    self.num_filters = num_filters
    self.l2_reg_lambda = l2_reg_lambda
    self.dropout_keep_prob = dropout_keep_prob


def inference(input_x, input_y, net_config):
  sequence_length = net_config.sequence_length
  num_classes = net_config.num_classes
  vocab_size = net_config.vocab_size
  embedding_size = net_config.embedding_size
  filter_sizes = net_config.filter_sizes
  num_filters = net_config.num_filters
  l2_reg_lambda = net_config.l2_reg_lambda
  dropout_keep_prob = net_config.dropout_keep_prob

  # Keeping track of l2 regularization loss (optional)
  l2_loss = tf.constant(0.0)

  # Embedding layer
  with tf.device('/cpu:0'), tf.variable_scope("embedding"):
    W = _variable_on_cpu('W',[vocab_size,embedding_size],
        tf.random_uniform_initializer( -1.0, 1.0))
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

  # Create a convolution + maxpool layer for each filter size
  pooled_outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv-maxpool-%s" % filter_size):
      # Convolution Layer
      filter_shape = [filter_size, embedding_size, 1, num_filters]
      W = _variable_on_cpu('W', filter_shape, tf.truncated_normal_initializer( stddev=0.1))
      b = _variable_on_cpu('b', num_filters,tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
      # Apply nonlinearity
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
      # Maxpooling over the outputs
      pooled = tf.nn.max_pool(
        h,
        ksize=[1, sequence_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
      pooled_outputs.append(pooled)

  # Combine all the pooled features
  num_filters_total = num_filters * len(filter_sizes)
  h_pool = tf.concat(3, pooled_outputs)
  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

  # Add dropout
  with tf.variable_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

  # Final (unnormalized) scores and predictions
  with tf.variable_scope("output"):
    W = _variable_on_cpu('W', [num_filters_total, num_classes], tf.truncated_normal_initializer(stddev=0.1))
    b = _variable_on_cpu('b', [num_classes], tf.constant_initializer(0.1))
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
    probs= tf.nn.softmax(scores,name="probs") 
  return predictions, scores, probs,l2_loss

def compute_loss(logits,labels, l2_loss, l2_reg_lambda):
  # CalculateMean cross-entropy loss
  with tf.variable_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits, tf.to_float(labels))
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
    tf.add_to_collection("losses", loss)

def accuracy(predictions, labels):
  # Accuracy
  with tf.variable_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
 

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def generate_batch(sentence, label, num_classes, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    sentence: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    sentences: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 2
  if shuffle:
    sentences, label_batch = tf.train.shuffle_batch(
        [sentence, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        enqueue_many=True,
        min_after_dequeue=min_queue_examples)
  else:
    sentences, label_batch = tf.train.batch(
        [sentences, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        nqueue_many=True,
        capacity=min_queue_examples + 3 * batch_size)

  return sentences, tf.reshape(label_batch, [batch_size,num_classes])
