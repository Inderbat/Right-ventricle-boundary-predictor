'''
	Joint Recognition of inner and outer boundary
'''
from train_utils import *

if __name__ == '__main__':

	# saving all the models
	save_models('loss_type')
	save_models('optimizer_type')
	save_models('learning_rate_scheduler')

	# creating the data loader for testing data
	data_loader_test_1 = load_data('test_set_1', dataset_split=None)
	data_loader_test_2 = load_data('test_set_2', dataset_split=None)

	# evaluating all the models
	model_evaluate(data_loader_test_1)
	model_evaluate(data_loader_test_2)
