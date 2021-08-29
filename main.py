import numpy
import matplotlib.pyplot as plt

xmin = 1
xmax = 10
x = numpy.linspace(xmin, xmax)

class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

def goalFunction(x):
    return x**(numpy.sin(10 * x))

def genAlgorithm():

