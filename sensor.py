import os, sys, pygame, math
from pygame.locals import *
from random import randint

class Sensor(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('media/sensor.png').convert()
        self.image.set_colorkey((255,255,255))
        self.rect = self.image.get_rect(center = (x,y))
        self.mask = pygame.mask.from_surface(self.image)
    
    def move(self,x,y):
        self.rect = self.image.get_rect(center = (x,y))
        self.mask = pygame.mask.from_surface(self.image)