# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
import sys
import time

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"mode": 'train',
	"data_dir": r'C:\Users\Administrator\Downloads\CSCE636-project-2022\cifar-10-python\cifar-10-batches-py'
	# ...
}

training_configs = {
	"learning_rate": 0.1,
	"batch_size": 128,
	"max_epoch": 300,
    "save_interval": 10,
	"modeldir": 'model_v1'
	# ...
}

### END CODE HERE