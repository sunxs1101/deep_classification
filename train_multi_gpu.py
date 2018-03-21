#-.- coding: utf-8 -.-
import tensorflow as tf
import numpy as np
import os
import time
import shutil
import datetime
import data_helpers
from text_cnn import TextCNN
import text_cnn

# Parameters
# ==================================================

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("max_steps", 20000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("early_stop_count", 100, "if accuracy not better for this time , stop training")
tf.flags.DEFINE_boolean("tmp_dir", False, "if True, write model result to ./runs/tmp/")
# Misc Parameters
"""
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
"""

tf.flags.DEFINE_string("data_path", "./data/data_server", "dir where training data are")
tf.flags.DEFINE_string("training_info", "default training setting", "any info differing this training from others")
flags = tf.app.flags
FLAGS = flags.FLAGS

train_info_dict={}

def tower_loss(scope,sentences,labels,net_config):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  _, logits, _ , l2_loss = text_cnn.inference(sentences, labels, net_config)
  
  text_cnn.compute_loss(logits, labels, l2_loss, net_config.l2_reg_lambda)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)
  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  """
  total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

# Training
# ==================================================
def train(data_set):
  print(FLAGS.training_info)
  #train_info_dict.append(FLAGS.training_info)
  print("\nParameters:")
  for attr, value in sorted(FLAGS.__flags.iteritems()):
      print("{}={}".format(attr.upper(), value))
      train_info_dict[attr]= value
  print("")
  record_train_info(train_info_dict, data_set) # record train info, vocabulory and label_inv  to file
  net_config= text_cnn.NetConfig(
       sequence_length=data_set.x_train.shape[1],
       num_classes=data_set.y.shape[1],
       vocab_size=len(data_set.vocabulary),
       embedding_size=FLAGS.embedding_dim,
       filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
       num_filters=FLAGS.num_filters,
       l2_reg_lambda=FLAGS.l2_reg_lambda,
       dropout_keep_prob=0.5)
  with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.AdamOptimizer(1e-4)
    # Calculate the gradients for each model tower.
    sentences,labels = text_cnn.generate_batch(data_set.x_train, data_set.y_train, net_config.num_classes, 16000, FLAGS.batch_size, True)
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (text_cnn.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss = tower_loss(scope,sentences,labels,net_config)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    #summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    """
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.histogram_summary(var.op.name + '/gradients', grad))
    """

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    """
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))
    """
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        text_cnn.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(),keep_checkpoint_every_n_hours=1)

    # Build the summary operation from the last tower summaries.
    #summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # Define Training procedure
        """
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """
        """
        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        """

        # Initialize all variables
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        # Generate batches
        # Training loop. For each batch...
        max_accuracy=0.0
        max_path=''
        max_count=0
        for step in xrange(FLAGS.max_steps):
          start_time = time.time()
          _, loss_value = sess.run([train_op, loss])
          duration = time.time() - start_time
    
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    
          if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus
    
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))
    
          # Save the model checkpoint periodically.
          if step % FLAGS.checkpoint_every == 0 or (step + 1) == FLAGS.max_steps:
            path = saver.save(sess, train_info_dict['checkpoint_prefix'], global_step=step)
            print("Saved model checkpoint to {}\n".format(path))
            """
            print(max_accuracy)
            if dev_accuracy >max_accuracy:
              max_accuracy = dev_accuracy
              max_count=1
              max_path=path
              shutil.copy(path,best_checkpoint_dir)
            else:
              max_count +=1
            if max_count > FLAGS.early_stop_count:
              print("early stop and best accuracy model at "+max_path)
              return 
            """
        print('end traing because hitting num_epochs {}'.format(FLAGS.num_epochs))

def record_train_info(train_info_dict, data_set):
  timestamp = str(int(time.time()))
  out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
  if FLAGS.tmp_dir: 
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs','tmp', timestamp))
  print("Writing to {}\n".format(out_dir))
  # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
  model_store_dir = os.path.abspath(os.path.join(out_dir, "model_store"))
  train_info_dict['model_dir']=model_store_dir #记录模型存放位置
  best_checkpoint_dir = os.path.abspath(os.path.join(model_store_dir, "best_checkpoints"))
  train_info_dict['best_checkpoint_dir']=best_checkpoint_dir
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  train_info_dict['checkpoint_prefix']=checkpoint_prefix
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  if not os.path.exists(model_store_dir):
    os.makedirs(model_store_dir)
  """write info to file"""
  model_info_dir=os.path.join(model_store_dir,'model_info')
  with open(model_info_dir,'w') as fmi:
    for k,v in train_info_dict.items():
      fmi.write('{}={}\n'.format(k,v))
  with open(os.path.join(model_store_dir,'label_inv'),'w') as f:
    for(k,v) in data_set.label_inv.items():
      f.write(" ".join((str(k),v)))
      f.write('\n')
  print("End write label-inv")
  with open(os.path.join(model_store_dir,"vocabulary"),'w') as f:
    for(k,v) in data_set.vocabulary.items():
      f.write(" ".join((k,str(v))))
      f.write('\n')
  print("End write vocabulary")
  return train_info_dict

if __name__ == '__main__':
  print('start loading data')
  data_set = data_helpers.DataSet(FLAGS.data_path)
  print('end loading data')
  print('waiting for 5 sec')
  time.sleep(5)
  train(data_set)
