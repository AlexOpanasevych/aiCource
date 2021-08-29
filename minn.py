import numpy as np
import random
import math
import matplotlib.pyplot as plt

BYTE_LENGTH = 10
LEFT_X = 1
RIGHT_X = 10


def maxValue(x):
    return 2 ** x - 1


def mapValue(value, left=LEFT_X, right=RIGHT_X):
    size = right - left
    pos = float(value) / maxValue(BYTE_LENGTH)
    return left + pos * size


def cross(a, b):
    pos = random.randint(0, BYTE_LENGTH - 1)
    c = a[:pos] + b[pos:]
    return c


def mutate(a, prob):
    for i in range(len(a)):
        prob_temp = prob / (len(a) - i)
        rand = random.random()
        if rand < prob_temp:
            if a[i] == 0:
                a[i] = 1
            else:
                a[i] = 0


def toDec(a):
    number = 0
    for i in range(1, len(a) + 1):
        number += 2 ** (i - 1) * a[-i]
    return mapValue(number)


def function(a):
    return math.pow(a, math.sin(10 * a))


def eval(a):
    val = toDec(a)
    return function(val)


def createPop(size):
    pop = []
    for i in range(size):
        creature = []
        for i in range(BYTE_LENGTH):
            c = random.randint(0, 1)
            creature.append(c)
        pop.append(creature)
    return pop


def select(_pop):
    for _ in range(int(len(_pop) / 2)):
        size = len(_pop)
        a1 = random.randint(0, size - 1)
        a2 = random.randint(0, size - 1)
        if eval(_pop[a1]) > eval(_pop[a2]):
            del _pop[a1]
        else:
            del _pop[a2]


def crossPop(pop, popSize):
    while len(pop) < popSize:
        a1 = random.randint(0, len(pop) - 1)
        a2 = random.randint(0, len(pop) - 1)
        pop.append(cross(pop[a1], pop[a2]))


def mutatePop(pop, prob):
    for creature in pop:
        mutate(creature, prob)


def evalPop(pop):
    best = eval(pop[0])
    bestCr = pop[0]
    avg = 0
    for creature in pop:
        val = eval(creature)
        if val < best:
            best = val
            bestCr = creature
        avg += val
    avg /= len(pop)
    print("Best val: " + str(best) + " best x: " + str(toDec(bestCr)) + " avg: " + str(avg))
    return best, avg


random.seed()
popSize = 30
mutationProb = 0.01
generations = 50
avgs = []
bests = []
pop = createPop(popSize)
bestVal, avg = evalPop(pop)
y_fun = []
x_fun = np.linspace(LEFT_X, RIGHT_X, 901)
for x in x_fun:
    y_fun.append(function(x))
for i in range(generations):
    print("\n--- generation " + str(i + 1) + " ---")
    select(pop)
    crossPop(pop, popSize)
    mutatePop(pop, mutationProb)
    best, avg = evalPop(pop)
    avgs.append(avg)
    bests.append(best)
    if best > bestVal:
        bestVal = best
    print("Best ever: " + str(bestVal))
    plotX = []
    plotY = []
    for creature in pop:
        plotX.append(toDec(creature))
        plotY.append(eval(creature))
    plt.figure()
    plt.plot(plotX, plotY, 'bo', x_fun, y_fun, 'k')
    plt.savefig('results/gen' + str(i + 1) + '.png')
    plt.close()
plt.figure()
plt.plot(avgs, 'k')
plt.savefig('results/averages.png')
plt.close()
