import os, sys, pygame, math
from pygame.locals import *
from random import randint


class Track(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('media/map2.png').convert()
        self.image.set_colorkey((255,255,255))
        CENTER_X =  int(pygame.display.Info().current_w /2)
        CENTER_Y =  int(pygame.display.Info().current_h /2)
        self.rect = self.image.get_rect(center = (CENTER_X,CENTER_Y))
        self.mask = pygame.mask.from_surface(self.image)
