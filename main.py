### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split,load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict", default= 'test')
parser.add_argument("--data_dir", help="path to the data", default= r'C:\Users\Administrator\Downloads\CSCE636-project-2022\starter_code\private_test_images_2022.npy')
parser.add_argument("--save_dir", help="path to save the results",default=r'C:\Users\Administrator\Downloads\CSCE636-project-2022')
parser.add_argument("--result_dir", help="path to save the results",default=r'C:\Users\Administrator\Downloads\CSCE636-project-2022\starter_code')

args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_valid, y_valid, [100,120,140,160,180,200,220,240,260,280,300],training_configs)


	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test, [100,120,140,160,180,200,220,240,260,280,300],training_configs)

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test, [300], training_configs)
		np.save(args.result_dir, predictions)
		

### END CODE HERE

