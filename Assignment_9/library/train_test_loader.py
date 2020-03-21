from imports_eva import *

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
train_dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
test_dataloader_args = dict(shuffle=False, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader

def get_train_test_loader(tr,ts):
	train_loader = torch.utils.data.DataLoader(tr, **train_dataloader_args)
	test_loader = torch.utils.data.DataLoader(ts, **test_dataloader_args)
	return train_loader,test_loader