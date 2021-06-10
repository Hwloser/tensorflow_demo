# import tensorflow lib
import tensorflow as tf

# tensor summary ()
# 1. 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
print(random_float)

print('-----------------------------------------------------')

# 2. 定义一个有两个元素的零向量
zero_vector = tf.zeros(shape=(2))
print(zero_vector)

print('-----------------------------------------------------')

# 3. 定义两个2x2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
print(A)
print(B)

print('-----------------------------------------------------')

'''
  张量的重要属性是其，形状、类型和值
'''
print(A.shape)
print(A.dtype)
'''
  [[1. 2.]
   [3. 4.]]
'''
print(A.numpy())

print('-----------------------------------------------------')


details = '''
  tensorflow operation
  
  add
  subtract
  multiplies
  
'''

print(details)

E = tf.add(A, B)
C = tf.subtract(A, B)
D = tf.matmul(A, B)

print(E)
print(C)
print(D)
