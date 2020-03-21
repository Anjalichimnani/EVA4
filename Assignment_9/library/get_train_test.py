from imports_eva import *
import data_loader

def get_train(db_name):
	return getattr(datasets, db_name)('./data', train=True, download=True, transform=albulmentation())
	
def get_test(db_name):
	return getattr(datasets, db_name)('./data', train=False, download=True, transform=albumentation_test())