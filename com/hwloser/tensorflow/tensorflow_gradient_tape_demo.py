import tensorflow as tf

initial_value_a = 3.
x = tf.Variable(initial_value=initial_value_a)

with tf.GradientTape() as tape:
    '''
      在 tf.GradientTape() 的上下文内，
      所有的计算步骤都会被巨鹿以用于求导
    '''
    y = tf.square(x)

y_grad = tape.gradient(y, x)  # y关于x的导数
print(y)
print(y_grad)
