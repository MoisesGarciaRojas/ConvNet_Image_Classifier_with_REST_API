# ConvNet_Image_Classifier_with_REST_API

Convolutional Neural Network which is trained based on the “mnist” dataset from Keras module. In addition, a REST API example is implemented.

# Description
As part of a job application at CollectAI, the challenge itself was to create an application, that could classify hand-written digits with a deep learning model. The code to build the model was provided by the company. What was required from me was to build a microservice that could expose REST endpoints for predictions and learning. A high prediction accuracy was not necessary.
It should be able to accept:
1) An image as an input for the prediction endpoint
2) A batch of images for the learning endpoint


## Python setup

1.	The “Project Interpreter” used is a 64 bits version of Python 3.6.
2.	The code was written on a 64 bits version of PyCharm Community Edition 2016.
3.	The following packages were used for the challenge:

### model.py

Package | Version
  ---   |   ---
Keras   | 2.2.2

### server.py

Package | Version
  ---   |   ---
numpy   | 1.15
aiohttp | 3.3.2

### client.py

Package       | Version
  ---      |   ---
aiohttp       | 3.3.2
asyncio       | xxx
async_timeout | 3.0.0
keras         | 2.2.2
