# Visualize neural networks using pygame
# Date: 4-9-20
# Resources:
# Flatten: https://www.w3resource.com/numpy/manipulation/ndarray-flatten.php
# Concatenate Numpy arrays: https://cmdlinetips.com/2018/04/how-to-concatenate-arrays-in-numpy/


import pygame
import numpy as np


class Network_architecture:
    def __init__(self, my_id):
        # Initialize the network architecture
        self.id = my_id                                 # Id of the network architecture
        self.input_layer = 4
        self.hidden_layers = [5,3]
        self.output_layer = 2
        self.num_weights, self.num_layers, self.num_nodes, self.cut_offs, self.matrix_dims = self.get_nums()
        self.weights = np.random.choice(np.arange(-1, 1, step=0.01), size=(1, self.num_weights), replace=True)
        print("num_weights: {}\nnum_layers: {}\tnum_nodes: {}\ncut_offs: {}\nmatrix_dims: {}\nweights: {}".format(
            self.num_weights, self.num_layers, self.num_nodes, self.cut_offs, self.matrix_dims, self.weights))
        self.weight_matrices = self.generate_matrices()     # Use this for forward propagation - will access same memory
        self.activations = None

    def get_nums(self):
        # Calculate the number of weights given the architecture.
        num_weights = self.input_layer * self.hidden_layers[0]      # Collect total number of weights
        num_layers = 2 + len(self.hidden_layers)                    # Get number of layers
        num_nodes = self.input_layer + self.output_layer            # Get number of nodes
        cut_offs = [0, num_weights]  # Where to start the cut offs for weight splicing
        matrix_dims = [(self.hidden_layers[0], self.input_layer)]   # List of dimensions out, in

        for i, layers in enumerate(self.hidden_layers):
            num_nodes += layers
            if i == num_layers - 3:    # On the last hidden layer
                num_weights += layers * self.output_layer
                matrix_dims.append((self.output_layer, layers))
            else:
                num_weights += layers * self.hidden_layers[i+1]
                matrix_dims.append((self.hidden_layers[i+1], layers))
            cut_offs.append(num_weights)    # Append the next cut off points

        return num_weights, num_layers, num_nodes, cut_offs, matrix_dims

    def generate_matrices(self):
        weight_matrices = []
        for i, cuts in enumerate(self.cut_offs):    # Go through all cut offs
            if i < len(self.cut_offs)-1:        # If not on the last index
                # Collect flattened sections of the weights list
                w_matrix = self.weights[0][cuts:self.cut_offs[i+1]]     # Splice the weight list
                reshaped_matrix = w_matrix.reshape(self.matrix_dims[i][0], self.matrix_dims[i][1])
                weight_matrices.append(reshaped_matrix)
        print(weight_matrices)
        return weight_matrices          # Different sized matrices - note they access the same memory as weights

    def softmax(self, z):
        # Softmax activation layer
        s = np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1, 1)
        return s

    def forward_propagation(self, state):
        self.activations = state[0].flatten()        # Activations all fall between -1 and 1 with tanh, start with input values[0]
        for i, matrix in enumerate(self.weight_matrices):
            if i == 0:  # input layer
                z_value = np.matmul(matrix, state.T)    # Multiply input by first weight matrix
                a_value = np.tanh(z_value)
                self.activations = np.concatenate((self.activations, a_value.T[0].flatten()))     # Combine transposed activation value

            elif i == len(self.weight_matrices) -1:     # The output layer
                z_value = np.matmul(matrix, a_value)
                a_value = self.softmax(z_value)
                self.activations = np.concatenate((self.activations, a_value.T.flatten()))    # Combine transposed activation value
            else:
                z_value = np.matmul(matrix, a_value)    # All hidden layers
                a_value = np.tanh(z_value)
                self.activations = np.concatenate((self.activations, a_value.T[0].flatten()))     # Combine transposed activation value

            print("z_value: {}".format(z_value))

        print("Activations: {}".format(self.activations))

        return a_value  # the final activation


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

    def weight_to_color(self, weight):
        # Takes a decimal between -1 and 1 weight and converts it to a color ratio between 0 and 255
        color_ratio = int((1 + weight)/2 * 255)
        return color_ratio

    def draw(self, window):

        # Show all nodes - optimize later
        layer_ind = 1                                               # Index of nodes per layer
        start_ind = self.npl[0]                                     # Starting node number
        end_ind = start_ind+self.npl[layer_ind]                     # End/cutoff node number
        next_layer_nodes = self.node_coords[start_ind: end_ind]     # Next layer's node coordinates
        weight_ind = 0

        # Display all the nodes
        for i, node in enumerate(self.node_coords):
            node_color_ratio = self.weight_to_color(self.network_arch.activations[i])
            pygame.draw.circle(window, (0, node_color_ratio, 255), node, self.node_radius)

            if i >= start_ind:     # if current node reaches next layer
                layer_ind += 1                  # Move index to next later
                if layer_ind < len(self.npl):   # Make sure index in range
                    start_ind = end_ind         # Update cut offs
                    end_ind += self.npl[layer_ind]      # Update end cut off
                    print("start: {}\t end: {}".format(start_ind, end_ind))
                    next_layer_nodes = self.node_coords[start_ind: end_ind]      # Update temp only when reach new layer
                    print(next_layer_nodes)

            if layer_ind < len(self.npl):  # Make sure index in range
                # Display the connections - ie the weights
                for next_nodes in next_layer_nodes:
                    color_ratio = self.weight_to_color(self.network_arch.weights[0][weight_ind])
                    pygame.draw.line(window, (255, color_ratio, 0), node, next_nodes, 1)      # Draw the connections between nodes
                    weight_ind += 1     # increment weight index
