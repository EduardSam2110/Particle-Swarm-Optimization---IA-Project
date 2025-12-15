from random import Random
import matplotlib.pyplot as plt

GLOBAL = 1
LOCAL = 0

class Particle:
    def __init__(self):
        self.x = 0 #current position of the particle
        self.v = 0 #velocity
        self.y = 0 #personal best position
        self.yn = 0 #neighborhood best position
        
def fitness(x):
    return x**4 + 16 

def calc_personal_best(p: Particle):
    if fitness(p.x) >= fitness(p.y): 
        return p.y
    else:
        return p.x

def calc_neighborhood_best(swarm):
    best = swarm[0].y
    for p in swarm:
        if fitness(p.y) < fitness(best):
            best = p.y
    return best

def calc_local_best(swarm, i):
    n = len(swarm)
    
    neighbors = [
        swarm[(i - 1) % n],
        swarm[i],
        swarm[(i + 1) % n]
    ]

    best = neighbors[0].y
    for p in neighbors:
        if fitness(p.y) < fitness(best):
            best = p.y
            
    return best


def velocity_update():
    pass

def PSO(mode):
    rand = Random()
    swarm = [Particle() for _ in range(10)]
    history = []

    for p in swarm:
        p.x = rand.uniform(-5, 5)
        p.v = 0
        p.y = p.x

    w, c1, c2 = 0.5, 1.5, 1.5

    for _ in range(50):
        
        if mode == GLOBAL:
            yn = calc_neighborhood_best(swarm)

        history.append([p.x for p in swarm])

        for i, p in enumerate(swarm):
            p.y = calc_personal_best(p)
            
            if mode == GLOBAL:
                p.yn = yn
            else:
                p.yn = calc_local_best(swarm, i)

            r1, r2 = rand.random(), rand.random()
            p.v = (w * p.v +
                   c1 * r1 * (p.y - p.x) +
                   c2 * r2 * (p.yn - p.x))
            p.x += p.v

    return swarm, history

if __name__ == "__main__":
    swarm, history = PSO(GLOBAL)

    xs = [i / 100 for i in range(-500, 501)]
    ys = [fitness(x) for x in xs]

    plt.ion()
    plt.figure()

    for step in history:
        plt.clf()
        plt.plot(xs, ys)
        plt.scatter(step, [fitness(x) for x in step])
        plt.pause(0.5)

    plt.ioff()
    plt.show()
