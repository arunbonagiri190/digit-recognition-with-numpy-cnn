import numpy as np

class Conv:

    filter_shape = (0,0)
    std_val = 9
    lr = 0.01
    

    def __init__(self, num_filters, filter_shape, lr):
        """
        Parameters:
            num_filters  : integer
            filter_shape : touple
            lr           : float
    
        """

        self.num_filters = num_filters
        self.filter_shape = filter_shape
        self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[1]) / self.std_val
        self.lr = lr


    def forward(self, input_image):
        """
        function for performing forward propagation

        Parameters:
            input_image : numpy array
        
        Returns:
            output : numpy array
        """
        
        self.last_input = input_image
        output = np.zeros((input_image.shape[0] - 2, input_image.shape[1] - 2, self.num_filters))

        for i in range(input_image.shape[0] - 2):
            for j in range(input_image.shape[1] - 2):
                
                selected_region = input_image[i:(i+self.filter_shape[0]), j:(j+self.filter_shape[1])]
                output[i, j] = np.sum(selected_region * self.filters, axis=(1, 2))

        return output


    def backprop(self, conv_error):
        """
        function for performing backward propagation

        Parameters:
            conv_error : numpy array
        
        Returns:
            (no-returns)
        """
        
        new_filters_weights = np.zeros(self.filters.shape)

        for i in range(self.last_input.shape[0] - 2):
            for j in range(self.last_input.shape[1] - 2):
                
                selected_region = self.last_input[i:(i+self.filter_shape[0]), j:(j+self.filter_shape[1])]
                for k in range(self.num_filters):
                    new_filters_weights[k] += conv_error[i, j, k] * selected_region

        self.filters -= self.lr * new_filters_weights

    # def get_output_shape(self, input_image):
    #     return np.zeros((input_image.shape[0] - 2, input_image.shape[1] - 2, self.num_filters)).shape