from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import sklearn.metrics as metrics
import warnings
import random
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Possible parameter values
min_c, max_c = 0.0, 1.0
kernels = ['rbf', 'poly', 'sigmoid', 'linear']
min_degree, max_degree = 2, 5
min_gamma, max_gamma = 0.0, 1.0
min_coef0, max_coef0 = 0.0, 1.0
min_iteration, max_iteration = 100, 400


logbook = []
train_x_arr = []
test_x_arr = []
train_Y = []
test_Y = []


def mutate(individual):

    gene = random.randint(0, 9)  # select which parameter to mutate
    if gene == 0:
        individual[0] = random.uniform(min_c, max_c)

    elif gene == 1:
        individual[1] = random.choice(kernels)

    elif gene == 2:
        individual[2] = random.randint(min_degree, max_degree)

    if gene == 3:
        individual[4] = random.uniform(min_gamma, max_gamma)

    if gene == 4:
        individual[4] = random.uniform(min_coef0, max_coef0)

    elif gene == 5:
        individual[5] = random.randint(min_iteration, max_iteration)

    return individual,


def evaluate(individual):
    '''
    build and test a model based on the parameters in an individual and return
    the Accuracy value
    '''
    # extract the values of the parameters from the individual chromosome
    c_ = individual[0]
    kernel_ = individual[1]
    degree_ = individual[2]
    gamma_ = individual[3]
    coef0_ = individual[4]
    iteration_ = individual[5]

    # build the model
    model_svm = SVC(C=c_,
                    coef0=coef0_,
                    degree=degree_,
                    gamma=gamma_, kernel=kernel_,
                    max_iter=iteration_,
                    probability=True
                    ).fit(train_x_arr, train_Y)
    predict = model_svm.predict(test_x_arr)

    accuracy_data = metrics.accuracy_score(predict, test_Y)*100
    accuracy = round(accuracy_data, 1)
    return accuracy,


def GA(population, crossover, mutation, generation):
  # Maximise the fitness function value
  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()

  N_CYCLES = 1

  toolbox.register("attr_c", random.uniform, min_c, max_c)
  toolbox.register("attr_kernel", random.choice, kernels)
  toolbox.register("attr_degree", random.randint, min_degree, max_degree)
  toolbox.register("attr_gamma", random.uniform, min_gamma, max_gamma)
  toolbox.register("attr_coef0", random.uniform, min_coef0, max_coef0)
  toolbox.register("attr_iteration", random.randint,
                   min_iteration, max_iteration)

  toolbox.register("individual", tools.initCycle, creator.Individual,
                   (toolbox.attr_c, toolbox.attr_kernel, toolbox.attr_degree,
                    toolbox.attr_gamma, toolbox.attr_coef0, toolbox.attr_iteration), n=N_CYCLES)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  toolbox.register("mate", tools.cxOnePoint)
  toolbox.register("mutate", mutate)
  toolbox.register("select", tools.selTournament, tournsize=2)
  toolbox.register("evaluate", evaluate)

  population_size = population
  crossover_probability = crossover
  mutation_probability = mutation
  number_of_generations = generation

  pop = toolbox.population(n=population_size)
  hof = tools.ParetoFront()
  # hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  print(stats)
  stats.register("avg", np.mean, axis=0)
  stats.register("std", np.std, axis=0)
  stats.register("min", np.min, axis=0)
  stats.register("max", np.max, axis=0)

  pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats=stats,
                                 mutpb=mutation_probability, ngen=number_of_generations, halloffame=hof,
                                 verbose=True)

  best_parameters = hof[0]  # save the optimal set of parameters
  global logbook
  logbook = log
  return log, best_parameters



def generatePlot():
    gen = logbook.select("gen")
    max_ = logbook.select("max")
    avg = logbook.select("avg")
    min_ = logbook.select("min")

    evolution = pd.DataFrame({'Generation': gen,
                            'Max Accuracy': max_,
                            'Average': avg,
                            'Min Accuracy': min_})

    plt.title('Parameter Optimisation')
    plot_ga = plt.plot(evolution['Generation'], evolution['Min Accuracy'], 'b', color = 'C1',
             label = 'Min')
    plot_ga = plt.plot(evolution['Generation'], evolution['Average'], 'b', color='C2',
             label = 'Average')
    plot_ga = plt.plot(evolution['Generation'], evolution['Max Accuracy'], 'b', color='C3',
            label='Max')


    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Generation')

    return plot_ga



def geneticAlgorithmProcess(train_x_arr_param, test_x_arr_param, train_Y_param, test_Y_param, population, crossover, mutation, generation):
    global train_x_arr, train_x_arr, test_x_arr, train_Y, test_Y
    train_x_arr = train_x_arr_param
    test_x_arr = test_x_arr_param
    train_Y = train_Y_param
    test_Y = test_Y_param
    log, best_param = GA(population, crossover, mutation, generation)
    plot_ga = generatePlot()
    return log, best_param, plot_ga
