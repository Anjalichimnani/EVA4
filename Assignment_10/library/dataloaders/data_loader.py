from imports.imports_eva import *

def get_train_transform(mu,sig):
  return transforms.Compose(
    [#transforms.RandomHorizontalFlip(),
     transforms.RandomRotation((-5.0, 5.0)),
     #transforms.RandomRotation(2),
     #transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),
     transforms.ColorJitter(hue=.05, saturation=.05),
     transforms.ToTensor(),
     transforms.Normalize(mu, sig)])
  
def get_test_transform(mu,sig):
  return transforms.Compose(
    [#transforms.RandomHorizontalFlip(),
     #transforms.RandomRotation((-5.0, 5.0)),
     #transforms.RandomRotation(2),
     #transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),
     #transforms.ColorJitter(hue=.05, saturation=.05),
     transforms.ToTensor(),
     transforms.Normalize(mu, sig)])