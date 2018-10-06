import numpy as np
import math
import time
import sys
import random
import pygame

PLANETS = [
    # (name, distance, velocity, mass, radius, sub-bodies)
    ("sun", 0, 0, 1.99e+30, 1.392e+9, []),
    ("mercury", 46e+9, 58.98e+3, 0.33011e+24, 2439.7e+3, []),
    ("venus", 107.48e+9, 35.26e+3, 4.8675e+24, 6051.8e+3, []),
    ("earth", 147.09e+9, 30.29e+3, 5.9723e+24, 6378.137e+3, [
        ("moon", 0.3633e+9, 1.082e+3, 0.07346e+24, 1738.1e+3, []),
    ]),
    ("mars", 206.62e+9, 26.5e+3, 0.64171e+24, 3396.2e+3, []),
    ("jupiter", 816.62e+9, 13.72e+3, 1898.19e+24, 71492e+3, []),
    ("saturn", 1352.55e+9, 10.18e+3, 568.34e+24, 60268e+3, []),
    ("uranus", 2741.30e+9, 7.11e+3, 86.813e+24, 25559e+3, []),
    ("neptune", 4444.45e+9, 5.5e+3, 102.413e+24, 24764e+3, []),
    ("pluto", 4436.82e+9, 6.1e+3, 0.01303e+24, 1187e+3, []),
]

SIZE = np.array([1000, 800])
GRAV = 6.67e-11
C = 299792458
FONT = None
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
        self.applied_force = np.zeros(2, dtype=np.float)
    
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
        
        self.applied_force = np.zeros(2, dtype=np.float)
        
    def coalesce(self, other):
        index = self.system.bodies.index(self)
        del self.system.bodies[index]
        self.system.following_index = None
    
    def discrete_step(self, dt):
        if self.system == None:
            return
                
        total_force = np.zeros(2, dtype=float)
        for body in self.system.bodies:
            if body == self:
                continue
            
            total_force += self.compute_force_to(body)
        
        total_force += self.applied_force
        
        self.acc = total_force / self.rmass()
        self.vel += self.acc * dt
        self.pos += self.vel * dt
    
    def apply_force(self, force):
        self.applied_force += force
    
    def required_timesteps(self):
         return max(round(1 / (1 - (self.speed()**2)/(C**2)) * self.system.dynamic_time_factor), self.system.time_resolution)

    def render(self, surface):
        vpos = ((self.pos + self.system.translation) / self.system.sf + SIZE / 2).astype(int)
        vrad = max(int(self.radius / (self.system.sf * self.system.radius_sf)), 0)

        self.trail.append((self.pos).astype(int).tolist())
        if len(self.trail) > 1:
            pygame.draw.lines(surface, (100, 100, 100), False, list(map(lambda p: [(p[0] + self.system.translation[0])/self.system.sf + SIZE[0]/2, (p[1] + self.system.translation[1])/self.system.sf + SIZE[1]/2], self.trail)), 1)
        if len(self.trail) > self.system.trail_len:
            self.trail = self.trail[1:]

        pygame.draw.circle(surface, (20, 20, 20, 30), vpos, int(self.schwarzchild_radius() / (self.system.sf * self.system.radius_sf)))
        pygame.draw.circle(surface, (255, 255, 255), vpos, vrad)
        label = FONT.render("%s (%fc)" % (self.name, math.hypot(self.vel[0], self.vel[1]) / 299792458), 1, (255, 255, 255))
        surface.blit(label, (vpos[0] + 1/SQRT_2*vrad, vpos[1] + 1/SQRT_2*vrad))

class System(object):
    def __init__(self):
        self.bodies = []
        self.last_frame = time.time()
        self.last_scaled_dt = 0

        self.following_index = 0
        self.trail_len = 500
        self.translation = np.array([0, 0], dtype=np.float)
        self.timescale_slowdown_exponent = 1.5
        self.dynamic_time_factor = 1
        self.time_resolution = 8
        self.timescale = 5e+4
        self.radius_sf = 1
        self.sf = 3e+8
    
    def add(self, body):
        body.system = self
        self.bodies.append(body)
    
    def construct_bodies(self, bodies, rel_pos=(0, 0), rel_vel=(0, 0)):
        for name, distance, velocity, mass, radius, sub_bodies in bodies:
            angle = random.random() * random.TWOPI
            x = distance * math.cos(angle) + rel_pos[0]
            y = -distance * math.sin(angle) + rel_pos[1]
            body = Body(name, x, y, mass, radius)
            body.vel[0] = -velocity * math.sin(angle) + rel_vel[0]
            body.vel[1] = -velocity * math.cos(angle) + rel_vel[1]
            self.add(body)
            self.construct_bodies(sub_bodies, rel_pos=(x, y), rel_vel=(body.vel[0], body.vel[1]))
    
    def step(self):
        dt = time.time() - self.last_frame
        self.last_frame = time.time()

        max_speed = max(map(lambda b: b.speed(), self.bodies)) / C

        for body in self.bodies:
            scaled_dt = min(dt * self.timescale / (max_speed ** self.timescale_slowdown_exponent), dt * self.timescale)
            body.step(scaled_dt)
            self.last_scaled_dt = scaled_dt
    
    def render(self, surface):
        for body in self.bodies:
            body.render(surface)
        
        following_name = "-" if self.following_index == None else self.bodies[self.following_index].name
        label = FONT.render("ts: %f; sts: %f; following %s" % (self.timescale, self.last_scaled_dt, following_name), 1, (0, 0, 0), (255, 255, 255))
        surface.blit(label, (0, 0))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def main():
    global FONT

    pygame.init()
    size = SIZE[0], SIZE[1]
    background = 0, 0, 0
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("pylanets")

    FONT = pygame.font.SysFont("monospace", 13)

    system = System()
    system.construct_bodies(PLANETS)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    system.following_index = (system.following_index + 1) % len(system.bodies) if system.following_index != None else 0
                elif event.key == pygame.K_a:
                    system.following_index = (system.following_index - 1) % len(system.bodies) if system.following_index != None else len(system.bodies)-1
                elif event.key == pygame.K_z:
                    system.following_index = None
                elif event.key == pygame.K_w:
                    system.timescale *= 1.1
                elif event.key == pygame.K_s:
                    system.timescale /= 1.1
                elif event.key == pygame.K_d:
                    system.radius_sf += 0.05
                elif event.key == pygame.K_e:
                    system.radius_sf = max(system.radius_sf - 0.05, 0.01)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    system.sf /= 1.1
                elif event.button == 5:
                    system.sf *= 1.1
        
        diff = pygame.mouse.get_rel()
        if pygame.mouse.get_pressed()[0]:
            system.translation[0] += diff[0] * system.sf
            system.translation[1] += diff[1] * system.sf
        elif system.following_index != None:
            wanted = np.array([
                -system.bodies[system.following_index].pos[0] - system.bodies[system.following_index].vel[0],
                -system.bodies[system.following_index].pos[1] - system.bodies[system.following_index].vel[1]
            ])
            
            system.translation += (wanted - system.translation) * 0.9

        system.step()
        
        screen.fill(background)
        system.render(screen)
        pygame.display.flip()

if __name__ == '__main__':
    main()
