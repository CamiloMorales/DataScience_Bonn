
# coding: utf-8

# In[1]:

import numpy as np
import operator
import time

import Individual, Helper, MLP


# In[1]:

def init(gp_config, train_set, val_set):
    
    def create_population():
        population = []
        for individual_number in range(gp_config["population-size"]):
            number_of_layers = np.random.randint(gp_config["min-layers"], gp_config["max-layers"] + 1)
            current_individual = []
            number_of_functions = len(gp_config["functions"])   
            for layer_number in range(number_of_layers*2):
                if(layer_number%2 == 0):
                    current_individual.append(np.random.randint(gp_config["min-units"], gp_config["max-units"] + 1))
                else:
                    current_individual.append(np.random.randint(0, number_of_functions))  
            new_individual = Individual.Individual()
            new_individual._individual = current_individual
            population.append(new_individual)
        return population
            
    def evaluate_fitness(start_from, population):
        count = 0
        for index in xrange(start_from, len(population)):
            print "Evaluating fitness " + str(index) + " of " + str(len(population))
            ind_array = population[index].individual
            layers = []
            for layer in xrange(start_from, len(ind_array), 2):
                layer_params = {}
                layer_params['num-units'] = ind_array[layer]
                layer_params['trans-func'] = gp_config['functions'][ind_array[layer+1]]
                layers.append(layer_params)
            mlp_basic_config = gp_config['MLP']
            mlp_basic_config['layer-defs'] = layers
            mlp = MLP.init(mlp_basic_config)
            mlp['train'](train_set)
            predicted = mlp['predict'](val_set['x'])
            #ind.fitness = Helper.estimate_cross_entropy(val_set['y'], predicted)
            population[index]._fitness = np.sum(np.multiply(val_set['y'],np.log(predicted + 0.0))) * (-1.0/val_set['y'].shape[0])       
            print "ind._fitness: "+str(population[index]._fitness)
            count+=1
        return population
    
    def do_cross_over(parent1, parent2):
        
        def do_cross_over(index, first_parent, second_parent):
            child = []
            for i in xrange(0, index):
                child.append(first_parent[i])
            for i in xrange(index, len(second_parent)):
                child.append(second_parent[i])
            new_individual = Individual.Individual()
            new_individual._individual = child
            return new_individual
        
        length = 0
        if len(parent1) < len(parent1):
            length = len(parent1)
        else:
            length = len(parent2)
        
        rand = np.random.randint(0, length - 1)
        
        if rand % 2 != 0:
            rand += 1
            
        children = []
        children.append(do_cross_over(rand, parent1, parent2))
        children.append(do_cross_over(rand, parent2, parent1))
        
        return children
    
    def do_mutation(individual):
        print "Do Mutation (length): " + str(len(individual.individual))
        rand = np.random.randint(0, len(individual.individual))
        if rand % 2 == 0:
            individual.individual[rand] = np.random.randint(gp_config['min-units'], gp_config['max-units'])
        else:
            individual.individual[rand] = np.random.randint(0, len(gp_config['functions']))
        
    
    def train():
        population = create_population()
        start_from = 0
        for step in xrange(0, gp_config['max-steps']):
            start_time = time.time()
            print "Training step: " + str(step)
            population = evaluate_fitness(start_from, population)
            population.sort(key=operator.attrgetter('fitness'))
            print 'step:' + str(step) + ' fitness:' +str(population[0].fitness) 
            for i in xrange(0, gp_config['population-size'] - gp_config['survival-size']):
                del population[-1]
            for i in xrange(0, int((gp_config['population-size'] - gp_config['survival-size'])/2)):
                p1 = population[np.random.randint(0, gp_config['survival-size'])]._individual
                p2 = population[np.random.randint(0, gp_config['survival-size'])]._individual
                children = do_cross_over(p1,p2)
                for child in children:
                    do_mutation(child)
                population += children
            start_from = gp_config['survival-size']
        population.sort(key=operator.attrgetter('fitness'))  
        print("Step: "+str(step)+ " took: " + str((time.time() - start_time)/60) + " minutes")
        return population[0]._individual
    return {
        'train':train
    }


# In[ ]:



