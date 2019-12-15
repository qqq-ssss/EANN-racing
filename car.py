import os, sys, pygame, math, trackmap, sensor
from pygame.locals import *
from random import randint
import torch
from torch import nn
import numpy as np


pygame.init()
CENTER_X =  int(pygame.display.Info().current_w /2)
CENTER_Y =  int(pygame.display.Info().current_h /2)
reset_dir = 90
reset_x = CENTER_X
reset_y = 220


def rot_center(image, rect, angle):
        """rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=rect.center)
        rot_mask = pygame.mask.from_surface(rot_image)
        return rot_image,rot_rect, rot_mask
    

def model_initialize():
        model = nn.Sequential()
        model.add_module('dense1', nn.Linear(5, 8))
        model.add_module('relu', nn.ReLU())
        model.add_module('dense2', nn.Linear(8, 6))
        model.add_module('relu', nn.ReLU())
        model.add_module('dense3', nn.Linear(6, 1))
        model.add_module('tanh', nn.Tanh())
        return model



#define car as Player.
class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('media/car_player_2.png').convert()
        self.image.set_colorkey((255,255,255))
        self.image_orig = self.image
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.x = reset_x
        self.y = reset_y
        self.rect.center = self.x, self.y
        self.dir = reset_dir
        self.speed = 0.0
        self.maxspeed = 4.0
        self.minspeed = -2.85
        self.acceleration = 0.01
        self.deacceleration = 0.12
        self.softening = 0.04
        self.steering = 15.0     
        self.sigma = 0.4
        self.circle = sensor.Sensor(self.x,self.y)
        self.model = model_initialize()
        self.off = False 
        self.chosen = False
        
        self.dir = reset_dir
        self.image, self.rect, self.mask = rot_center(self.image_orig, self.rect, self.dir)
        
        self.fwd = 0
        self.l_fwd = 0
        self.r_fwd = 0
        self.left = 0
        self.right = 0
        
        self.fwd_dot = 0
        self.l_fwd_dot = 0
        self.r_fwd_dot = 0
        self.left_dot = 0
        self.right_dot = 0
        
        self.fwd_flag = 0
        self.l_fwd_flag = 0
        self.r_fwd_flag = 0
        self.left_flag = 0
        self.right_flag = 0
    
    def swap_gene(self, gene1, gene2, gene3):
        self.model.dense1.weight=torch.nn.Parameter(torch.Tensor(gene1 + np.random.normal(loc=0.0, scale=self.sigma, size=[8,5])))
        self.model.dense2.weight=torch.nn.Parameter(torch.Tensor(gene2 + np.random.normal(loc=0.0, scale=self.sigma, size=[6,8])))
        self.model.dense3.weight=torch.nn.Parameter(torch.Tensor(gene3 +np.random.normal(loc=0.0, scale=self.sigma, size=6)))
    
    def save(self):
        torch.save(self.model.state_dict(), 'model.pth')
        
    def load(self):
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()
        self.model.dense1.weight=torch.nn.Parameter(self.model.dense1.weight+torch.Tensor(np.random.normal(loc=0.0, scale=self.sigma, size=[8,5])))
        self.model.dense2.weight=torch.nn.Parameter(self.model.dense2.weight+torch.Tensor(np.random.normal(loc=0.0, scale=self.sigma, size=[6,8])))
        self.model.dense3.weight=torch.nn.Parameter(self.model.dense3.weight+torch.Tensor(np.random.normal(loc=0.0, scale=self.sigma, size=6)))
        
    def switch_off(self):
        self.off = True
        self.speed = 0.0
        
    def switch_on(self):
        self.off = False
        
    def choose(self, position_x, position_y):
        self.circle.move(position_x, position_y)
        if pygame.sprite.collide_mask(self,self.circle) is not None and not self.chosen:            
            self.chosen = True
            genes1 = self.model.dense1.weight.detach().numpy()
            genes2 = self.model.dense2.weight.detach().numpy()
            genes3 = self.model.dense3.weight.detach().numpy()
            self.image = pygame.image.load('media/car_player_chosen.png').convert()
            self.image.set_colorkey((255,255,255))
            self.image_orig = self.image
            self.rect = self.image.get_rect()
            self.mask = pygame.mask.from_surface(self.image)
            self.rect.center = self.x, self.y
            self.image, self.rect, self.mask = rot_center(self.image_orig, self.rect, self.dir)
            return (genes1, genes2, genes3)
        else:
            return None
        
    def reset(self):
        self.x = reset_x
        self.y = reset_y
        self.speed = 0.0
        self.dir = reset_dir
        self.chosen = False
        self.image = pygame.image.load('media/car_player_2.png').convert()
        self.image.set_colorkey((255,255,255))
        self.image_orig = self.image
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect.center = self.x, self.y
        self.image, self.rect, self.mask = rot_center(self.image_orig, self.rect, self.dir)
        
        
    def sensors(self, racemap, angle):        
        if (self.dir<=angle):
            fwd_dir = angle-self.dir
        else:
            fwd_dir = 360+angle-self.dir    
            
        flag = True
            
        i = 0
        while (flag) & (i<100):
            x_dot = math.floor(self.x + i*math.cos(math.radians(fwd_dir)))
            y_dot = math.floor(self.y + i*math.sin(math.radians(fwd_dir)))
            self.circle.move(x_dot,y_dot)
            i+=5
            if pygame.sprite.collide_mask(racemap,self.circle) is not None:            
                flag = False
                
        dist = round(((self.x - x_dot)**2 + (self.y - y_dot)**2)**0.5,1)
        dot = (x_dot,y_dot)
                               
        return dist, dot, flag        


    def soften(self):
            if self.speed > 0:
                self.speed -= self.softening
            if self.speed < 0:
                self.speed += self.softening

    def accelerate(self):
        if self.speed < self.maxspeed:
            self.speed = self.speed + self.acceleration
    
    def nn_accelerate(self, acc):
        if self.speed < self.maxspeed:
            self.speed = self.speed + self.acceleration*(acc+1.5)
            
    def nn_steer(self, steer):
        self.dir = self.dir+steer*self.steering
        if self.dir > 360:
            self.dir = 0
        if self.dir < 0:
            self.dir = 360
        self.image, self.rect, self.mask = rot_center(self.image_orig, self.rect, self.dir)
            

    def deaccelerate(self):
        if self.speed > self.minspeed:
            self.speed = self.speed - self.deacceleration


    def steerleft(self):
        self.dir = self.dir+self.steering
        if self.dir > 360:
            self.dir = 0
        self.image, self.rect, self.mask = rot_center(self.image_orig, self.rect, self.dir)


    def steerright(self):
        self.dir = self.dir-self.steering
        if self.dir < 0:
            self.dir = 360
        self.image, self.rect, self.mask = rot_center(self.image_orig, self.rect, self.dir)


    def update(self, racemap):
        self.x = self.x + self.speed * math.cos(math.radians(270-self.dir))
        self.y = self.y + self.speed * math.sin(math.radians(270-self.dir))
        self.rect.center = self.x, self.y
        
        self.fwd, self.fwd_dot, self.fwd_flag = self.sensors(racemap, 270)
        self.l_fwd, self.l_fwd_dot, self.l_fwd_flag = self.sensors(racemap, 225)
        self.left, self.left_dot, self.left_flag = self.sensors(racemap,180)
        self.r_fwd, self.r_fwd_dot, self.r_fwd_flag = self.sensors(racemap, 315)
        self.right, self.right_dot, self.right_flag = self.sensors(racemap,360)
        self.fwd -= 8
        self.right -= 5
        self.left -=5
        self.r_fwd -= 7
        self.l_fwd -= 7