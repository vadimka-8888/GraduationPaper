
def CreateScheduler1(min_lr, max_lr, num_epoch):
    step_1 = int(0.3 * num_epoch)
    step_2 = int(0.7 * num_epoch)
    def Scheduler_1(epoch, lr):
        if epoch < step_1:
            return 1e-3
        elif epoch >= step_1 and epoch < step_2:
            return 5e-4
        else:
            return 1e-4

def CreateScheduler2(min_lr, max_lr, num_epoch):
    def Scheduler_2(epoch, lr):



lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

def CreateCosineScheduler(min_lr, max_lr, num_epoch):
    cycle_length = num_epoch - 10
    def SchedulerCosine(epoch, lr):
        #min_lr = 1e-4
        #max_lr = 1e-1

    	if epoch <= cycle_length:
    		unit_cycle = (1 + math.cos(iteration * math.pi / cycle_length)) / 2
    		adjusted_cycle = (unit_cycle * (max_lr - min_lr)) + min_lr
    		return adjusted_cycle
    	else:
    		return min_lr

	return SchedulerCosine

def CreateTriangularScheduler(min_lr, max_lr, num_epoch):
    inc_fraction=0.5
	def TriangularScheduler(epoch, lr)
        if iteration <= cycle_length * inc_fraction:
            unit_cycle = iteration * 1 / (cycle_length * inc_fraction)
        elif iteration <= cycle_length:
            unit_cycle = (cycle_length - iteration) * 1 / (cycle_length * (1 - inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (max_lr - min_lr)) + min_lr
        result.append(adjusted_cycle)
    return result