import pygame
import car, trackmap, math, torch
import numpy as np
                          
def main():
    clock = pygame.time.Clock()
    player_pool = []
    for i in range(50):
        player_pool.append(car.Player())
    racemap = trackmap.Track()
    font = pygame.font.Font(None, 24)
    game_flag = True
    draw_sensor = True
    printgene_flag = False
    new_epoch = False
    load_model = False
    x_pressed = False
    chosenmax = 1
    chosen = 0
    epoch = 1
    genelist1 = []
    genelist2 = []
    genelist3 = []
    saving_time = 0
    loading_time = 0

    while game_flag:
        
        if saving_time>0:
            saving_time -= 1
            
        if loading_time>0:
            loading_time -= 1
        
        if new_epoch:
            new_epoch = False
            epoch += 1
            genesum1 = np.zeros([8,5])
            for i in genelist1:
                genesum1 = genesum1+i
            genesum1/=chosen
            genesum2 = np.zeros([6,8])
            for i in genelist2:
                genesum2 = genesum2+i
            genesum2/=chosen
            genesum3 = np.zeros(6)
            for i in genelist3:
                genesum3 = genesum3+i
            genesum3/=chosen
            chosen = 0
            genelist1 = []
            genelist2 = []
            genelist3 = []
            for player in player_pool:
                player.reset()
                player.switch_on()
                if not load_model:
                    player.swap_gene(genesum1,genesum2, genesum3)
                else:
                    player.load()
                    
                    
            load_model = False
                    
            
        choose_flag = False
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                game_flag = False
            if i.type == pygame.MOUSEBUTTONDOWN:
                if i.button == 1:
                    position_x = i.pos[0]
                    position_y = i.pos[1]
                    choose_flag = True
                    
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            game_flag = False
        if keys[pygame.K_n] & (chosen>0):
            new_epoch = True
        if keys[pygame.K_s] & (chosen==1):
            saving_time = 64
            for player in player_pool:
                if player.chosen:
                    player.save()
        if keys[pygame.K_l]:
            new_epoch = True
            load_model = True
            loading_time = 64   
        if keys[pygame.K_x]:
            
            if not x_pressed:
                if draw_sensor:
                    draw_sensor = False
                else:
                    draw_sensor = True
            x_pressed = True
        else:
            x_pressed = False
            
            
        screen.fill((190,190,190))
        screen.blit(racemap.image, racemap.rect)    
            
        for player in player_pool:
            if not player.off:
                X = torch.tensor([player.left, player.l_fwd, player.fwd, player.r_fwd, player.right], dtype = torch.float32)
                steering = player.model(X)
                player.nn_steer(steering.detach().numpy())
                player.nn_accelerate(1)
            
            
                player.update(racemap)
        
                if pygame.sprite.collide_mask(player,racemap) is not None:
                    player.switch_off()
                
                if draw_sensor:
                    flag_list = [(player.fwd_flag, player.fwd_dot),
                                 (player.l_fwd_flag, player.l_fwd_dot),
                                 (player.r_fwd_flag, player.r_fwd_dot),
                                 (player.right_flag, player.right_dot),
                                 (player.left_flag, player.left_dot)]
                    for flag in flag_list:
                        if flag[0]:
                            pygame.draw.circle(screen, (160,0,24), flag[1], 3)
                        else:
                            pygame.draw.circle(screen, (156,156,39), flag[1], 3)
            
            if choose_flag:
                gene =  player.choose(position_x, position_y)
                if gene is not None:
                    choose_flag = False
                    genelist1.append(gene[0])
                    genelist2.append(gene[1])
                    genelist3.append(gene[2])
                    chosen+=1
                    
            
            screen.blit(player.image, player.rect)
        
            
                
        text_fps = font.render('FPS: ' + str(int(clock.get_fps())), 1, (124, 64, 37))
        text_chosen = font.render('Chosen: ' + str(chosen), 1, (124, 64, 37))
        text_epoch = font.render('Epoch: ' + str(epoch), 1, (124, 64, 37))
        text_sensors = font.render('Drawing sensors: ' + str(draw_sensor), 1, (124, 64, 37))
        if (saving_time>0)&(loading_time>0):
            text_saving = font.render('Model saved', 1, (124, 64, 37))
            screen.blit(text_saving, (400,400))
            text_loading = font.render('Model loaded', 1, (124, 64, 37))
            screen.blit(text_loading, (400,450))
        elif loading_time>0:
            text_loading = font.render('Model loaded', 1, (124, 64, 37))
            screen.blit(text_loading, (400,400))
        elif saving_time>0:
            text_saving = font.render('Model saved', 1, (124, 64, 37))
            screen.blit(text_saving, (400,400))
            
            
            
        screen.blit(text_fps, (400,200))
        screen.blit(text_chosen, (400,250))
        screen.blit(text_epoch, (400,300))
        screen.blit(text_sensors, (400,350))
        pygame.display.update()
        
        clock.tick(64)
         
    

pygame.init()

screen = pygame.display.set_mode((pygame.display.Info().current_w,
                                  pygame.display.Info().current_h),
                                  pygame.FULLSCREEN)

pygame.display.set_caption('NNRacing')
pygame.mouse.set_visible(True)

CENTER_W =  int(pygame.display.Info().current_w /2)
CENTER_H =  int(pygame.display.Info().current_h /2)

main()

pygame.quit()