########################################################################################################################
# File_name:            server.py                                                                                      #
# Creator:              Collect AI                                                                                     #
# Created:              Tuesday - August 7, 2018                                                                       #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Monday - August 20, 2018                                                                       #
# Description:          Import of class ImageClassifier. The welcome, prediction, and learning handlers, as well as    #
#                       the creation of the server application where added for the purpose of the challenge.           #
#                       The prediction handler, receives the image in a serial format and it transform the data to     #
#                       be used in the "predict" method. The learning handler receives the images in a dictionary      #
#                       form (labels, attributes), these are passed to the "train_new" method in a serial format       #
########################################################################################################################

from model.model import ImageClassifier
import numpy as np
from aiohttp import web

# example usage of the model class
cls = ImageClassifier()
cls.open_model()
# cls.train()
# cls.save_model()
# print(cls.predict(cls.x_test))

# Add a "welcome" handler to test the web application is up and running
# and two other handlers for learning and prediction methods
async def welcome(request):
    return web.Response(text="Hello, keras and other packages are loading and the web app\n"
                             "wait until server has fully started", status=200)

# Prediction handler
async def prediction(request):
    try:
        # Receive and format data to be suitable for prediction method
        data = await request.post()
        values = np.array(list(data.values()))
        values = values.reshape(1,28, 28, 1)
        values = values.astype('float32')
        values_scl = values/255

        # Get prediction
        pred = cls.predict(values_scl)
        pred = pred*100
        text = "Predicted digit: " + str(np.asscalar(pred.argmax()))

        # Return predicted value
        return web.Response(text=text, status=200)

    # Error handler
    except Exception as e:
        response = {'Status': 'failed', 'Message': str(e)}
        return web.Response(text=response, status=500)

# New trained model
async def learning(request):
    try:
        # Receive and format data to be suitable for training method
        batch = await request.post()
        labels = batch["labels"]
        labels = eval(labels)
        labels = np.array(list(labels.values()))
        attributes = batch["attributes"]
        attributes = eval(attributes)
        attributes = np.array(list(attributes.values()))

        # Train new model
        cls.train_new(labels, attributes)

        # Return message of completion
        return web.Response(text="Training finished")

    # Error handler
    except Exception as e:
        response = {'Status': 'failed', 'Message': str(e)}
        return web.Response(text=response, status=500)

# Create application and load handlers for prediction and learning
if __name__ == "__main__":
    app = web.Application(client_max_size=10485760) # Max size 10MB
    app.router.add_get('/', welcome)
    app.router.add_post('/prediction', prediction)
    app.router.add_post('/learning', learning)
    web.run_app(app)