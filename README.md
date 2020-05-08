# SenKuu —— Deep Learning for beginners
![head](./other/pics/head.png)

![LatestVersion](https://img.shields.io/badge/LatestVersion-0.1.0-blue.svg)   

**NOTE：** 可能会由于网络问题无法正常显示图片，请clone到本地查看

## Welcome
基于 NumPy 的深度学习模型开发框架，通过搭积木的方式来组装神经网络模型。API 借鉴了 Keras 的设计。

**Contact Me**: senkuu @ 163.com

## License
![license](https://img.shields.io/badge/license-Apache-brightgreen.svg)  
<br/>
LearnDL is distributed under the Apache license 2.0.

## Getting started: 10 seconds to SenKuu
``` python
from senkuu.model import Model
from senkuu.structures.layers import Input, Dense

model = Model()
model.add(Input(units=2))  # The first layer must be Input layer
model.add(Dense(units=3, activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))

model.set(loss='binary_crossentropy', optimizer='adam',  
          metrics=['acc', 'precision', 'recall', 'f1'])

model.train(train_x, train_y, epochs=100, validation=0.2)

loss, score = model.test(test_x, test_y)
print(loss, score)

model.predict(new_x, onlyclass=True)
```

## Installation
There are two ways to install Senkuu:  
* (**Recommended**) install SenKuu from PyPI:  
``` shell
$ pip install senkuu
```

* (**Alternatively**) install SenKuu from GitHub source:
``` shell
$ git clone https://github.com/techrc/senkuu.git
```

## Developed & Developing Functions
![functions](./other/pics/functions.png)

## Architecture
![architecture](./other/pics/architecture.png)

## Data Flow
![data flow](./other/pics/dataflow.png)

## Why this name, SenKuu?
![senkuu](./otherpics/senkuu.png)  
Here is Senkuu, senkuu from **《Dr.Stone》**
