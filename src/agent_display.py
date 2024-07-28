import pygame as pg
import torch
import numpy as np
class InformationDisplay:

    #see https://highway-env.farama.org/actions/
    ACTION_2_STR = {
        0 : "LEFT",
        1 : "IDLE",
        2 : "RIGHT",
        3 : "ACC",
        4 : "DEC",
    }

    grey = (105,105,105)
    white = (255, 255, 255)
    green = (0, 105, 0)

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    def display_meta_information(self, agent_display: pg.Surface, sim_display: pg.Surface) -> None:
        white = InformationDisplay.white
        grey = InformationDisplay.grey

        #print(pg.font.get_fonts())
        X = agent_display.get_width()
        Y = agent_display.get_height()
        header_font = pg.font.SysFont('dejavuserif', 24)
        font = pg.font.SysFont('dejavuserif', 18)
        agent_display.fill(grey)

        ####### ACTION UTILITIES #######
        text = header_font.render('Action utilities', True, white, grey)
        textRect = text.get_rect()
        textRect.left = 10
        textRect.top = 10
        agent_display.blit(text, textRect)

        self.print_action_utilities(agent_display, left=10, right= X//2,top= 50, bottom= Y, font=font)

        ####### OBSERVER VEHICLE INFORMATION #######
        text = header_font.render('Obs. vehicle info', True, white, grey)
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

        #speed of observer vehicles
        curr_speed = self.env.unwrapped.controlled_vehicles[0].speed
        target_speed = self.env.unwrapped.controlled_vehicles[0].target_speed

        #curr_speed
        text = font.render(f'Vehicle speed: {curr_speed:.2f}', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 55
        textRect.top = 110
        agent_display.blit(text, textRect)

        #target speed
        text = font.render(f'Target speed: {target_speed:.2f}', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 60
        textRect.top = 140
        agent_display.blit(text, textRect)

        ####### ALL VEHICLE INFORMATION #######
        text = header_font.render('All vehicle info', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 70
        textRect.top = 200
        agent_display.blit(text, textRect)

        #speed summary of all vehicles
        vehicle_speeds = np.array([v.speed for v in self.env.unwrapped.road.vehicles])
        mean_speed = vehicle_speeds.mean()
        std_speed = vehicle_speeds.std()

        #mean speed
        text = font.render(f'Mean speed: {mean_speed:.2f}', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 60
        textRect.top = 230
        agent_display.blit(text, textRect)

        #std speed
        text = font.render(f'Std speed: {std_speed:.2f}', True, white, grey)
        textRect = text.get_rect()
        textRect.right = X - 60
        textRect.top = 260
        agent_display.blit(text, textRect)

    def print_action_utilities(self, display: pg.Surface, left: int, right: int, top: int, bottom: int, font: pg.font):
        white = InformationDisplay.white
        grey = InformationDisplay.grey
        action_utilities = np.full(shape=(5,), fill_value=0)
        if hasattr(self.agent, "action_utility_values"):
            action_utilities = self.agent.action_utility_values
        
        action_x_dir_space = (right - left) // action_utilities.shape[0]
        action_y_dir_space = (bottom - top) // 2
        max_action = action_utilities.argmax()
        for i in range(action_utilities.shape[0]):
            value_center_x = action_x_dir_space * i + action_x_dir_space//2

            #header_text
            text = font.render(f'{InformationDisplay.ACTION_2_STR[i]}', True, white, grey)
            textRect = text.get_rect()
            textRect.center = (value_center_x, top)
            display.blit(text, textRect)

            #value_text
            text = font.render(f'{action_utilities[i]:.3f}', True, white)
            textRect = text.get_rect()
            textRect.center = (value_center_x, action_y_dir_space)

            if i == max_action:
                temp_surface = pg.Surface(text.get_size())
                temp_surface.fill(InformationDisplay.green)
                temp_surface.blit(text, (0, 0))
                surface_rect = temp_surface.get_rect()
                surface_rect.center = (value_center_x, action_y_dir_space)
                display.blit(temp_surface, surface_rect)
            else:
                display.blit(text, textRect)

            


