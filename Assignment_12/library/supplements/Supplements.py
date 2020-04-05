from imports.imports_eva import *

class Supplements:
    def __init__():
        super().__init__()
		
    def show_training_schedule(lr_min, lr_max, train_set_size, batch_size, factor_step_iteration, number_of_cycles):
        lr = []
        iteration_plot = []

        iterations = math.ceil(train_set_size/batch_size)
        stepsize = factor_step_iteration * iterations
        total_iterations = number_of_cycles * iterations

        for i in range(total_iterations):
            cycle = np.floor(1 + i/(2 * stepsize))
            x = np.abs(i/stepsize - 2*cycle + 1)
            lr_value = lr_min + (lr_max - lr_min) * (1 - x)
            lr.append (lr_value)
            iteration_plot.append(i)

        plt.figure(figsize= (20, 3))
        plt.plot (iteration_plot, lr)
        plt.show()