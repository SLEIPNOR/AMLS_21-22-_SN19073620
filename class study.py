# class Circle():  # 创建Circle类，Circle为类名
#    pass  # 此处可添加属性和方法
# #%%
# circle1= Circle()
# circle2= Circle()
#
# circle1.r = 1
# circle2.R = 2
#%%
class Circle():
   pi = 0.34 # 类属性
   def __init__(self, R):
      self.r = R # same as circle1.r = 1

circle1 = Circle(1)#  实例创建 (R)
print(circle1.r)
circle2 = Circle(2)
print(circle2.r)

#%%

circle1 = Circle(1)
circle2 = Circle(2)
print('----未修改前-----')
print('pi=\t', Circle.pi)
print('circle1.pi=\t', circle1.pi)  #  3.14
print('circle2.pi=\t', circle2.pi)  #  3.14
print('----通过类名修改后-----')
Circle.pi = 3.14159  # 通过类名修改类属性，所有实例的类属性被改变
print('pi=\t', Circle.pi)   #  3.14159
print('circle1.pi=\t', circle1.pi)   #  3.14159
print('circle2.pi=\t', circle2.pi)   #  3.14159

print('----通过circle1实例名修改后-----')
circle1.pi=3.14111   # 实际上这里是给circle1创建了一个与类属性同名的实例属性
print('pi=\t', Circle.pi)     #  3.14159
print('circle1.pi=\t', circle1.pi)  # 实例属性的访问优先级比类属性高，所以是3.14111
print('circle2.pi=\t', circle2.pi)  #  3.14159
print('----删除circle1实例属性pi-----')
# 可见，千万不要在实例上修改类属性，它实际上并没有修改类属性，而是给实例绑定了一个实例属性。
#%% python 类的实例方法
class Circle(object):
   pi = 3.14

   def __init__(self, R):
      self.radius = R
   def get_area(self):
      return self.radius**2*self.pi

circle1 = Circle(1)

print(circle1.get_area())

circle2 = Circle(6)

print(circle2.get_area())

#%%
# from tensorflow.keras import layers,activations,models
# model = models.Sequential()
# model.add(layers.Conv2D(filters=3,kernel_size=3, padding='same', strides=1, input_shape=(6, 6, 3), activation='relu'))
# model.add(layers.Conv2D(filters=3,kernel_size=3, padding='same', activation='relu'))
# # 这里其实还有一层是将输入数据和现在的层的输出加了一下，只不过不影响输出形状，可以通过自定义层实现，这里就不写了
# model.add(layers.BatchNormalization())
# model.summary()

# import tensorflow as tf
# from tensorflow.keras import layers
#
# net = tf.keras.models.Sequential(
#     [layers.Conv2D(64, kernel_size=7, strides=2, padding='same'
#                    , input_shape=(224, 224, 1), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
#
# net.summary()


class A:
    def add(self, x):
        y = x + 1
        print(y)


class B(A):
    def add(self, x):

        super().add(x)


b = B()
b.add(2)  # 3

a = A()
a.add(2)