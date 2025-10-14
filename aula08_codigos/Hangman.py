from numpy.random import randint


geneset = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
target = "Hello World!"
generation = 1

def generateParent(length):
    genes = list("")
    for i in range(0,length):
        geneIndex = randint(0, len(geneset));
        genes.append(geneset[geneIndex])
    return(''.join(genes))

def getFitness(candidate, target):
   fitness = 0
   for i in range(0, len(candidate)):
       if target[i] == candidate[i]:
           fitness += 1
   return(fitness)

def mutate(parent):
   geneIndex = randint(0, len(geneset));
   index = randint(0, len(parent))
   genes = list(parent)
   genes[index] = geneset[geneIndex]
   return(''.join(genes))

def display(candidate):
    fitness = getFitness(candidate, target)
    print ("%s,  fitness = %i generations = %s" %(candidate, fitness, generation))
    
bestParent = generateParent(len(target))
bestFitness = getFitness(bestParent, target)
display(bestParent)
 
while bestFitness < len(bestParent):
    generation = generation +1
    child = mutate(bestParent)
    childFitness = getFitness(child, target)
 
    if childFitness > bestFitness:
      bestFitness = childFitness
      bestParent = child
      display(bestParent)