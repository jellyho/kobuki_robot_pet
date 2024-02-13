import numpy

def wrap_to_pi(x):
    return numpy.mod(x+numpy.pi,2*numpy.pi)-numpy.pi

def sign(x):
    if x > 0: return +1
    if x < 0: return -1
    return 0