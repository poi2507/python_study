# 케라스 창시자에게 배우는 딥러닝
## 1장, 2장 정리 mnist예제

1. **1장 딥러닝이란 무엇인가?**
  - **인공지능** <br> "컴퓨터가 '생각'할 수 있는가?"라는 질문을 하면서 시작되었다.  보통 사람이 수행하는 지능적인 작업을 자동화 하기 위한 연구 활동
  - **심볼릭 AI** <br> 인간 수준의 인공 지능을 만들 수 있다고 믿었다. 체스게임처럼 잘 정의된 논리적인 문제를 푸는 데 적합하다는 것이 증명되었지만  복잡하고 불분명한 문제를 해결하기 어려움(머신러닝의 등장 이유)
  - **머신러닝** <br>
  우리가 어떤 것을 작동시키기 위해 '어떻게 명령할 지 알고 있는 것'이상을 컴퓨터가 처리하는 것이 가능한가?, 특정 작업을 수행하는 법을 스스로 학습할 수 있는가?
  머신러닝 시스템은 명시적으로 프로그램되는 것이 아니라 **훈련** 된다. 머신러닝에서 **학습** 이란 더 나은 표현을 찾는 자동화된 과정
  - **데이터에서 표현을 학습하기** <br>
  머신러닝은 샘플과 기댓값이 주어졌을 때 데이터 처리 작업을 위한 실행 규칙을 찾는 것 입니다. <br>
    * 입력 데이터 포인트<br>
    * 기대출력<br>
    * 알고리즘의 성능을 측정하는 방법<br>
머신 러닝과 딥러니의 핵심 문제는 **의미 있는 대이터로의 변환** 입력데이터를 기반으로 기대 출력에 가깝게 만드는 유용한 표련을 학습하는 것이다.
  - **딥러닝** <br>
  딥러닝은 머신러닝의 특정한 한 분야로서 연속된 층에서 점진적으로 의미 있는 표현을 배우는 강점이 있으며, 데이터로 부터 표현을 학습하는 새로은 방식<br>
  데이터로부터 모델을 만드는 데 얼마나 많은 층을 사용했는지가 그 모델의 깊이가 된다. 딥러닝은 그냥 데이터로부토 표현을 학습하는 수학 모델이다.<br>
  딥러닝은 특성 공학의 단계를 완전히 자동화 한다.<br>
  딥러닝이 데이터로 부터 학습하는 방법에는 두가지 중요한 특징이 있다. **층을 거치면서 점진적으로 더 복잡한 표현이 만들어진다** , **이런 점진적인 중간 표현이 공동으로 학승된다** 이 두가지 특징으로 인래 머신러닝 접근 방법보다
  딥러닝이 성공하게된 이유이다.
  ---
1. **2장 시작하기 전에: 신경망의 수학적 구성요소**  
  - 신경망을 위한 데이터 표현 <br>
  텐서 : 다차원 넘파이 배열에 데이터를 저장하는 것, 데이터를 위한 컨테이너 <br>
    * 스칼라(0D텐서) : 하나의 숫자만 담고 있는 텐서, 축의 개수 = 0
    * 벡터(1D 텐서) : 숫자의배열, 축의 개수 = 1
    * 행렬(2D 텐서) : 벡터의 배열, 축의 개서 = 2 <br>
  핵심속성에는 **축의 개수**, **크기**, **데이터 타입** 이 있다
<br>
1. **mnist예제** <br>

<pre>
In [1]:
from keras.datasets import mnist
Using TensorFlow backend.
In [36]:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
In [37]:
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255
In [38]:
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255
In [48]:
train_labels.shape
Out[48]:
(60000,)
In [39]:
train_images.shape
Out[39]:
(60000, 784)
In [40]:
len(train_labels)
Out[40]:
60000
In [41]:
train_labels
Out[41]:
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
In [42]:
from keras import models
from keras import layers
In [54]:
network = models.Sequential()
In [55]:
network.add(layers.Dense(512,activation = 'relu',input_shape = (28*28,)))
network.add(layers.Dense(10,activation = 'softmax'))
In [61]:
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
In [56]:
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics=['accuracy'])
In [62]:
network.fit(train_images,train_labels,epochs = 5, batch_size = 128)
WARNING:tensorflow:From /home/poi2507/.conda/envs/deep-learning-with-python/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/5
60000/60000 [==============================] - 6s 107us/step - loss: 0.2577 - acc: 0.9257
Epoch 2/5
60000/60000 [==============================] - 6s 103us/step - loss: 0.1041 - acc: 0.9691
Epoch 3/5
60000/60000 [==============================] - 6s 104us/step - loss: 0.0683 - acc: 0.9793
Epoch 4/5
60000/60000 [==============================] - 6s 103us/step - loss: 0.0505 - acc: 0.9843
Epoch 5/5
60000/60000 [==============================] - 6s 102us/step - loss: 0.0382 - acc: 0.9886
Out[62]:
<keras.callbacks.History at 0x7f37071e7b00>
In [63]:
test_loss,test_acc = network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)
10000/10000 [==============================] - 1s 52us/step
test_acc: 0.9798
</pre>
