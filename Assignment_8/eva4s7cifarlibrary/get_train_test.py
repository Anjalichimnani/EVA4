from all_imports import *
import data_loader

def get_train(db_name):
	return getattr(datasets, db_name)('./data', train=True, download=True, transform=data_loader.get_train_transform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
	
def get_test(db_name):
	return getattr(datasets, db_name)('./data', train=False, download=True, transform=data_loader.get_test_transform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))