########################################################################################################################
# File_name:            client.py                                                                                      #
# Creator:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Tuesday - August 7, 2018                                                                       #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Monday - August 20, 2018                                                                       #
# Description:          Client example to test the REST API, it loads again the "mnist" dataset. The connection is     #
#                       first established, then "send_digit" method sends one digit to predict and "send_batch"        #
#                       sends several images to train a new model. The client session is initialized in the last       #
#                       four lines                                                                                     #
########################################################################################################################

import aiohttp
import asyncio
import async_timeout
from keras.datasets import mnist

NUM_IMGS = 256
IDX_IMAGE = 3

# Try minst data again for training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Fetch localhost
async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

# Create ClientSession object
async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://localhost:8080/')
        print(html)

# Send one image for prediction (image is in matrix form)
async def send_digit():

    # Print label of test image on Python console
    print("Test image is the digit: " + str(y_test[IDX_IMAGE]))

    # Serialize attributes
    params = {}
    rng = len(x_test[IDX_IMAGE].flatten())
    for i in range(rng):
        params[i] = x_test[IDX_IMAGE].flatten()[i]

    # Send serialized data and wait for response from the server
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8080/prediction', data=params) as resp:
            print(await resp.text())

# Send several images to train new model (images are in matrix form)
async def send_batch():
    # Print reference message on Python console
    print("New data for training has been sent, wait until receiving a completion message")

    # Take from 0 to N images
    labels = y_train[0:NUM_IMGS]

    # Serialize labels as a dictionary
    param_lab = {}
    rng = len(labels)
    for i in range(rng):
        param_lab[i] = labels.flatten()[i]
    attributes = x_train[0:NUM_IMGS]
    attributes = attributes.flatten()

    # Serialize attributes as a dictionary
    param_att = {}
    for j in range(len(attributes)):
        param_att[j] = attributes.flatten()[j]
    # Create a dictionary of dictionaries
    batch = {'labels': str(param_lab), 'attributes': str(param_att)}

    # Send serialized data and wait for response from the server
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8080/learning', data=batch) as resp:
            print(await resp.text())

# Create loop event for the application
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.run_until_complete(send_digit())
loop.run_until_complete(send_batch())