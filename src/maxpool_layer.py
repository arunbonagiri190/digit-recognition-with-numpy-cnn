import numpy as np


class MaxPool:

    image_shape = [0, 0]
    num_filters = 0
    
    def forward(self, input_image):
        """
        function for performing forward propagation

        Parameters:
            input_image : numpy array
        
        Returns:
            output : numpy array
        """

        self.last_input = input_image
        self.image_shape[0], self.image_shape[1], self.num_filters = input_image.shape
        
        output = np.zeros(((self.image_shape[0]//2), (self.image_shape[1]//2), self.num_filters))
        for i in range((self.image_shape[0] // 2)):
            for j in range((self.image_shape[1] // 2)):
                
                selected_region = input_image[(i*2):(i*2+2),(j*2):(j*2+2)]
                output[i, j] = np.amax(selected_region, axis=(0, 1))
        
        return output

    def backprop(self, error):
        """
        function for performing backward propagation

        Parameters:
            error : numpy array
        
        Returns:
            conv_error : numpy array
        """
    
        conv_error = np.zeros(self.last_input.shape)

        for i in range(self.last_input.shape[0]//2):
            for j in range(self.last_input.shape[1]//2):
                
                selected_region = self.last_input[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                h, w, f = selected_region.shape
                amax = np.amax(selected_region, axis=(0, 1))
        
        for i2 in range(h):
            for j2 in range(w):
                for f2 in range(f):
                    # If this pixel was the max value, copy the gradient to it.
                    if selected_region[i2, j2, f2] == amax[f2]:
                        conv_error[i * 2 + i2, j * 2 + j2, f2] = error[i, j, f2]

        return conv_error

    # def get_output_shape(self):
    #     return np.zeros(((self.image_shape[0]//2), (self.image_shape[1]//2), self.num_filters)).shape
