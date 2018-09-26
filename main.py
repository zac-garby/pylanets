import numpy as np
import math
import time
import sys
import pygame

PLANETS = [
    # (name, distance, velocity, mass, diameter)
    ("sun", 0, 0, 1.99e+30, 1.392e+9),
    ("mercury", 6.98e+10, 47.4e+3, 3.3e+23, 4.878e+6),
    ("venus", 1.089e+11, 35.0e+3, 4.87e+24, 1.2104e+7),
    ("earth", 1.496e+11, 29.8e+3, 5.97e+24, 1.2756e+7),
    ("mars", 2.2379e+11, 24.1e+3, 6.42e+23, 6.794e+6),
    ("jupiter", 7.786e+11, 13.1e+3, 1.9e+27, 1.42984e+8),
    ("saturn", 1.433e+12, 9.6e+3, 5.69e+26, 1.20536e+8),
    ("uranus", 2.873e+12, 6.8e+3, 8.68e+25, 5.1118e+7),
    ("neptune", 4.495e+12, 5.4e+3, 1.03e+26, 4.9528e+7),
    ("pluto", 5.906e+12, 4.74e+3, 1.46e+22, 2.37e+5)
]

GRAV = 6.67e-11
SF = 5e+8
RAD_SF = 1.5e-2
TIMESCALE = 9e+5
TV = np.array([400, 400], dtype=np.float)
TRAIL_LEN = 500

class Body(object):
    def __init__(self, name, x, y, mass, radius):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.system = None
        self.trail = []

        self.pos = np.array([x, y], dtype=np.float)
        self.vel = np.array([0, 0], dtype=np.float)
        self.acc = np.array([0, 0], dtype=np.float)
    
    def compute_force_to(self, other):
        dist = math.hypot(other.pos[0] - self.pos[0], other.pos[1] - self.pos[1])
        magnitude = (GRAV * self.mass * other.mass) / (dist ** 2)

        force = np.array([
            other.pos[0] - self.pos[0],
            other.pos[1] - self.pos[1],
        ], dtype=np.float)

        return normalize(force) * magnitude
    
    def step(self, dt):
        if self.system == None:
            return
        
        self.acc = np.zeros(2, dtype=float)
        
        for body in self.system.bodies:
            if body == self:
                continue
            
            force = self.compute_force_to(body)
            self.acc += force / self.mass
        
        self.vel += self.acc * dt
        self.pos += self.vel * dt

    def render(self, surface):
        vpos = (self.pos / SF + TV).astype(int)
        vrad = max(int(self.radius / (SF * RAD_SF)), 0)

        self.trail.append((self.pos).astype(int).tolist())
        if len(self.trail) > 1:
            pygame.draw.lines(surface, (100, 100, 100), False, list(map(lambda p: [p[0]/SF + TV[0], p[1]/SF + TV[1]], self.trail)), 1)
        if len(self.trail) > TRAIL_LEN:
            self.trail = self.trail[1:]

        pygame.draw.circle(surface, (255, 255, 255), vpos, vrad)

class System(object):
    def __init__(self):
        self.bodies = []
        self.last_frame = time.time()
    
    def add(self, body):
        body.system = self
        self.bodies.append(body)
    
    def step(self):
        dt = time.time() - self.last_frame
        self.last_frame = time.time()

        for body in self.bodies:
            body.step(dt * TIMESCALE)
    
    def render(self, surface):
        for body in self.bodies:
            body.render(surface)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def main():
    global TV
    global SF

    pygame.init()
    size = width, height = 800, 800
    background = 0, 0, 0
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("pylanets")

    #earth = Body("earth", 0, AU, EARTH_MASS, EARTH_RAD)
    #sun = Body("sun", 0, 0, SUN_MASS, SUN_RAD)

    #earth.vel[0] = 29800

    #system = System()
    #system.add(earth)
    #system.add(sun)

    system = System()

    for name, distance, velocity, mass, diameter in PLANETS:
        planet = Body(name, 0, distance, mass, diameter/2)
        planet.vel[0] = velocity
        system.add(planet)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    SF /= 1.1
                elif event.button == 5:
                    SF *= 1.1
        
        diff = pygame.mouse.get_rel()
        if pygame.mouse.get_pressed()[0]:
            TV[0] += diff[0]
            TV[1] += diff[1]

        system.step()
        
        screen.fill(background)
        system.render(screen)
        pygame.display.flip()

if __name__ == '__main__':
    main()