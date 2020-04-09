# Visualize neural networks

import pygame
import numpy as np


class Network_architecture:
    def __init__(self, my_id):
        # Initialize the network architecture
        self.id = my_id
        self.input_layer = 4
        self.hidden_layers = [5,5]
        self.output_layer = 2
        self.num_weights, self.num_layers, self.num_nodes = self.get_nums()
        print(self.num_weights, self.num_layers, self.num_nodes)

    def get_nums(self):
        # Calculate the number of weights given the architecture.
        num_weights = self.input_layer * self.hidden_layers[0]
        num_layers = 2 + len(self.hidden_layers)
        num_nodes = self.input_layer + self.output_layer
        for i, layers in enumerate(self.hidden_layers):
            num_nodes += layers
            if i == num_layers - 3:    # On the last hidden layer
                num_weights += layers * self.output_layer
            else:
                num_weights += layers * self.hidden_layers[i+1]

        return num_weights, num_layers, num_nodes


class Visual:
    def __init__(self, network_arch, screen_w, screen_h):
        self.node_radius = 10
        self.network_arch = network_arch
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x_divs, self.y_divs, self.npl = self.get_divs()  # Get those dividers and nodes per layer
        self.node_coords = self.get_coords()
        print("coords: {}\nnpl: {}".format(self.node_coords, self.npl))

    def get_divs(self):     # Get divider coordinates
        x_coords_div = int(self.screen_w/(self.network_arch.num_layers + 1))
        nodes_per_layer = [self.network_arch.input_layer]                           # Nodes per layer
        y_coords_div =[(int(self.screen_h/(self.network_arch.input_layer + 1)))]    # append input layer
        for i in range(len(self.network_arch.hidden_layers)):
            nodes_per_layer.append(self.network_arch.hidden_layers[i])              # Nodes in hidden layer
            y_coords_div.append(int(self.screen_h/(self.network_arch.hidden_layers[i] + 1)))    # add hidden layers

        nodes_per_layer.append(self.network_arch.output_layer)                          # Nodes in output layer
        y_coords_div.append(int(self.screen_h/(self.network_arch.output_layer + 1)))    # append output layer

        return x_coords_div, y_coords_div, nodes_per_layer

    def get_coords(self):   # get the coordinates given the dividers
        coords = []
        for i in range(len(self.npl)):  # Go through all nodes per layers
            x = self.x_divs * (i + 1)   # x coordinate
            for j in range(self.npl[i]):
                y = self.y_divs[i] * (j + 1)    # y coordinate
                coords.append((x, y))

        return coords       # Return list of coordinate tuples



    def draw(self, window):

        # Show all nodes - optimize later
        j = 1  # Index of nodes per layer
        start_ind = self.npl[0]
        end_ind = start_ind+self.npl[j]
        for i, node in enumerate(self.node_coords):
            pygame.draw.circle(window, (0, np.random.randint(0,255), 200), node, self.node_radius, self.node_radius)

            if i >= start_ind:     # if current node reaches next layer
                j += 1                  # Move index to next later
                if j < len(self.npl):   # Make sure index in range
                    start_ind = end_ind         # Update cut offs
                    end_ind += self.npl[j]      # Update end cut off
                    print("start: {}\t end: {}".format(start_ind, end_ind))

            if j < len(self.npl):   # Make sure index in range
                temp = self.node_coords[start_ind: end_ind]      # Next layer's nodes
                print(temp)

                for next_nodes in temp:
                    pygame.draw.line(window, (255, np.random.randint(0,255), 0), node, next_nodes, 1)      # Draw the connections between nodes

