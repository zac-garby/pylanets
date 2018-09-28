import numpy as np
import math
import time
import sys
import pygame

PLANETS = [
    # (name, distance, velocity, mass, radius)
    ("sun", 0, 0, 1.99e+30, 1.392e+9),
    ("mercury", 46e+9, 58.98e+3, 0.33011e+24, 2439.7e+3),
    ("venus", 107.48e+9, 35.26e+3, 4.8675e+24, 6051.8e+3),
    ("earth", 147.09e+9, 30.29e+3, 5.9723e+24, 6378.137e+3),
    ("mars", 206.62e+9, 26.5e+3, 0.64171e+24, 3396.2e+3),
    ("jupiter", 816.62e+9, 13.72e+3, 1898.19e+24, 71492e+3),
    ("saturn", 1352.55e+9, 10.18e+3, 568.34e+24, 60268e+3),
    ("uranus", 2741.30e+9, 7.11e+3, 86.813e+24, 25559e+3),
    ("neptune", 4444.45e+9, 5.5e+3, 102.413e+24, 24764e+3),
    ("pluto", 4436.82e+9, 6.1e+3, 0.01303e+24, 1187e+3),
    ("moon", 147.09e+9 + 0.3633e+9, 30.29e+3 + 1.082e+3, 0.07346e+24, 1738.1e+3),
]

GRAV = 6.67e-11
C = 299792458
SF = 3e+8
RAD_SF = 1
TIMESCALE = 5e+3
DYNAMIC_TIME_FACTOR = 1
TIMESCALE_SLOWDOWN_EXPONENT = 1.5
TV = np.array([0, 0], dtype=np.float)
TRAIL_LEN = 500
FONT = None
FOLLOWING = None
SQRT_2 = math.sqrt(2)

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
        dist = self.dist(other)
        magnitude = (GRAV * self.rmass() * other.mass) / (dist ** 2)

        force = np.array([
            other.pos[0] - self.pos[0],
            other.pos[1] - self.pos[1],
        ], dtype=np.float)

        return normalize(force) * magnitude
    
    def dist(self, other):
        return math.hypot(other.pos[0] - self.pos[0], other.pos[1] - self.pos[1])
    
    def rmass(self):
        try:
            return self.mass / math.sqrt(1 - math.pow(self.speed(), 2) / math.pow(C, 2))
        except Exception as e:
            print(self.mass, self.speed(), self.speed() / C)
            raise e
    
    def speed(self):
        return math.hypot(self.vel[0], self.vel[1])
    
    def schwarzchild_radius(self):
        return (2 * self.rmass() * GRAV) / (C ** 2)
    
    def step(self, dt):
        for body in self.system.bodies:
            if body == self:
                continue
            
            if self.dist(body) < body.schwarzchild_radius() + self.schwarzchild_radius():
                if self.rmass() > body.rmass():
                    body.coalesce(self)
                else:
                    self.coalesce(other)

        timesteps = self.required_timesteps()
        cdt = dt / timesteps

        for i in range(timesteps):
            self.discrete_step(cdt)
        
    def coalesce(self, other):
        global FOLLOWING

        index = self.system.bodies.index(self)
        del self.system.bodies[index]
        FOLLOWING = None
    
    def discrete_step(self, dt):
        if self.system == None:
            return
                
        total_force = np.zeros(2, dtype=float)
        for body in self.system.bodies:
            if body == self:
                continue
            
            total_force += self.compute_force_to(body)
        
        self.acc = total_force / self.rmass()
        self.vel += self.acc * dt
        self.pos += self.vel * dt
    
    def required_timesteps(self):
         return round(1 / (1 - (self.speed()**2)/(C**2)) * DYNAMIC_TIME_FACTOR)

    def render(self, surface):
        vpos = ((self.pos + TV) / SF + 400).astype(int)
        vrad = max(int(self.radius / (SF * RAD_SF)), 0)

        self.trail.append((self.pos).astype(int).tolist())
        if len(self.trail) > 1:
            pygame.draw.lines(surface, (100, 100, 100), False, list(map(lambda p: [(p[0] + TV[0])/SF + 400, (p[1] + TV[1])/SF + 400], self.trail)), 1)
        if len(self.trail) > TRAIL_LEN:
            self.trail = self.trail[1:]

        pygame.draw.circle(surface, (20, 20, 20, 30), vpos, int(self.schwarzchild_radius() / (SF * RAD_SF)))
        pygame.draw.circle(surface, (255, 255, 255), vpos, vrad)
        label = FONT.render("%s (%fc)" % (self.name, math.hypot(self.vel[0], self.vel[1]) / 299792458), 1, (255, 255, 255))
        surface.blit(label, (vpos[0] + 1/SQRT_2*vrad, vpos[1] + 1/SQRT_2*vrad))

class System(object):
    def __init__(self):
        self.bodies = []
        self.last_frame = time.time()
        self.last_scaled_dt = 0
    
    def add(self, body):
        body.system = self
        self.bodies.append(body)
    
    def step(self):
        dt = time.time() - self.last_frame
        self.last_frame = time.time()

        max_speed = max(map(lambda b: b.speed(), self.bodies)) / C

        for body in self.bodies:
            scaled_dt = min(dt * TIMESCALE / (max_speed ** TIMESCALE_SLOWDOWN_EXPONENT), dt * TIMESCALE)
            body.step(scaled_dt)
            self.last_scaled_dt = scaled_dt
    
    def render(self, surface):
        for body in self.bodies:
            body.render(surface)
        
        following_name = "-" if FOLLOWING == None else self.bodies[FOLLOWING].name
        label = FONT.render("ts: %f; sts: %f; following %s" % (TIMESCALE, self.last_scaled_dt, following_name), 1, (0, 0, 0), (255, 255, 255))
        surface.blit(label, (0, 0))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def main():
    global TV
    global SF
    global FONT
    global FOLLOWING

    pygame.init()
    size = width, height = 800, 800
    background = 0, 0, 0
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("pylanets")

    FONT = pygame.font.SysFont("monospace", 13)

    system = System()

    for name, distance, velocity, mass, radius in PLANETS:
        planet = Body(name, 0, distance, mass, radius)
        planet.vel[0] = velocity
        system.add(planet)

    FOLLOWING = 10

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
            TV[0] += diff[0] * SF
            TV[1] += diff[1] * SF
        elif FOLLOWING != None:
            wanted = np.array([
                -system.bodies[FOLLOWING].pos[0],
                -system.bodies[FOLLOWING].pos[1]
            ])
            
            TV += (wanted - TV) * 0.3

        system.step()
        
        screen.fill(background)
        system.render(screen)
        pygame.display.flip()

if __name__ == '__main__':
    main()