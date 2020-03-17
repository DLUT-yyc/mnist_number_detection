#### mnist_number_detection

##### Introduction：

Use opencv to conditionally limit the number box to get its coordinate position, and input the detected target part into the trained forward propagation network to complete the entire target detection process.

##### requirements:

ubuntu(recommend 18.04 or 16.04)

ROS(ros-melodic-desktop-full is ok)

pytorch(with cuda is best)

venv(you need to use python3 in the ros environment which has a bad support for python3)

##### Quick Start

pytorch_mnist.py is the code to train the mnist

![train](C:\Users\19162\github\1.jpg)
pytorch_mnist_camera.py is the code to detect and classify the number you write

![1584415415715](C:\Users\19162\github\2.jpg)

![1584415434958](C:\Users\19162\github\3.jpg)

Others are just for learning.
