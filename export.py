import tensorflowjs as tfjs

def train(model):

    tfjs.converters.convert(model, "data")

train("data/celsius_fahrenhenit.h5")