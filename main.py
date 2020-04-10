# Main file where the magic happens
# Date: 4-9-20

import numpy as np
import pygame
import visualize as vs


# Game class to manage game settings
class Game:
    def __init__(self, screen_w, screen_h, num_networks):
        pygame.display.set_caption('Network Visualizer')     # Caption
        self.screen_w = screen_w                # Screen width
        self.screen_h = screen_h                # Screen height
        self.window = pygame.display.set_mode((screen_w, screen_h))  # Game window
        self.crash = False                      # When to shut down the game

        self.net_archs = vs.Network_architecture(0)
        self.vis = vs.Visual(self.net_archs, self.screen_w, self.screen_h)

    def run(self):
        '''
        Run the game to show network.
        :return:
        '''
        clock = pygame.time.Clock()
        frames = 0

        while not self.crash:                   # Keep going while the game hasn't ended.
            clock.tick(30)                      # Frames per second
            for event in pygame.event.get():    # Get game close event - if user closes game window
                if event.type == pygame.QUIT:
                    self.crash = True           # Crash will get us out of the game loop

            move = np.random.choice(np.arange(0, 2, step=1), size=(1,4), replace=True)
            print("Softmax Activation: {}".format(self.net_archs.forward_propagation(move)))
            # Draw everything on screen once per frame
            self.draw_window(frames)

            frames += 1

    def draw_window(self, frames):
        self.window.fill((0, 0, 0))  # Screen Color fill
        # Draw stuff here
        self.vis.draw(self.window)

        pygame.display.update()


if __name__ == '__main__':
    print("VISUALIZE NEURAL NETWORKS")

    game = Game(800, 600, 1)
    game.run()

