import tensorflow as tf

c0 = tf.constant([0.1],dtype=tf.float32)
c1 = tf.constant([5.0],dtype=tf.float32)

cBool =  tf.greater(c1,c0)
if tf.greater(c1,c0):
  cTmp = tf.constant([1.2])
else:
  cTmp = tf.constant([1.6])
c3=cTmp

with tf.Session() as sess1:
  print(sess1.run(cBool))
  print(True==sess1.run(cBool[0]))
  print(sess1.run(c3))
  

