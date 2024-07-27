import pygame as pg
import torch
class InformationDisplay:

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent


    def display_meta_information(self, agent_display: pg.display, sim_display: pg.display) -> None:
        #print(pg.font.get_fonts())
        X = agent_display.get_width()
        Y = agent_display.get_height()
        grey = (105,105,105)
        white = (255, 255, 255)
        header_font = pg.font.SysFont('dejavuserif', 24)
        font = pg.font.SysFont('dejavuserif', 18)
        agent_display.fill(grey)

        #Action utilities
        text = header_font.render('Action utilities', True, white, grey)
        textRect = text.get_rect()
        textRect.left = 10
        textRect.top = 10
        agent_display.blit(text, textRect)

        #Action q value estimates
        text = header_font.render('Action Q-values', True, white, grey)
        textRect = text.get_rect()
        textRect.left = 10
        textRect.top = Y//2
        agent_display.blit(text, textRect)

        ####### EPISODE INFORMATION #######
        text = header_font.render('Episode information', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 40
        textRect.top = 10
        agent_display.blit(text, textRect)

        #objective weights
        obj_weights = torch.round(self.agent.objective_weights, decimals=2)
        text = font.render(f'Obj. Weights: ({obj_weights[0]:.2f}, {obj_weights[1]:.2f})', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 10
        textRect.top = 50
        agent_display.blit(text, textRect)

        #num controlled vehicles
        
        text = font.render(f'Controlled vehicles: {self.agent.num_controlled_vehicles}', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 40
        textRect.top = 80
        agent_display.blit(text, textRect)


