#For data loading and processing
from torchvision.transforms import Compose,Resize,Normalize,ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import math

#For model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#For shwing images and graphs
from matplotlib import pyplot as plt

#For evaluation
from sklearn.metrics import confusion_matrix

#For generic torch functions
import torch

#For Evaluation
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#For loading/unloading
import pickle

#For demo
import sys

imagePath = 'Data/classified/' #Change here
mean=[0.485, 0.456, 0.406]
std=[0.225, 0.225, 0.225]
test_ratio = 0.2
validation_ratio = 0.2

transformer = Compose([Resize((128,128)), ToTensor(), Normalize(mean=mean,std=std)])
data = ImageFolder(root=imagePath, transform=transformer)
loader = DataLoader(data,batch_size=32)

# Defining Test Validation & Train sizes
test_size = math.floor(test_ratio*len(data))
train_size = len(data) - test_size
validation_size = math.floor(validation_ratio*train_size)
train_size -= validation_size


class ConvNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers =  nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.FC = nn.Sequential(
            nn.Linear(128*16*16,50),
            nn.Linear(50,4)
            
        )
    
    def forward(self,X):
        X = self.convlayers(X)
        X =  self.FC(X.reshape(-1,128*16*16))
        return X

if __name__ == '__main__':
	try:
		print('Running {} in {} mode'.format(sys.argv[0],sys.argv[1]))
		mode = sys.argv[1]
	except:
		print('Did you pass enough arguements? Try again with correct arguements.')

		print('train_size:{}, validation_size:{}, test_size:{}, total:{}'
			.format(train_size,validation_size,test_size,train_size+validation_size+test_size))

		# Loaders with split
	torch.manual_seed(0)
	shuffledIndices = torch.randperm(len(data))
	train_loader = DataLoader(data,batch_size=32,sampler=shuffledIndices[:train_size])
	validation_loader = DataLoader(data,batch_size=32,sampler=shuffledIndices[train_size:train_size+validation_size])
	test_loader = DataLoader(data,batch_size=32,sampler=shuffledIndices[train_size+validation_size:])

	if mode=='train':
		cnn = ConvNet5().cuda()
		optimizer = optim.Adam(cnn.parameters(),lr=0.00001)
		loss_func = nn.CrossEntropyLoss()
		epochs = 1
		training_losses = []
		validation_losses = []

		print('Starting training loop.')
		for e in range(epochs, epochs+3):
		    cnn.train()
		    training_loss=0
		    for i, (batch,labels) in enumerate(train_loader):
		        y_h = cnn(batch.cuda())
		        cnn.zero_grad()
		        training_loss = loss_func(y_h,labels.cuda())
		        training_loss.backward()
		        optimizer.step()
		    training_losses.append(training_loss)
		    
		    #'''
		    cnn.eval()
		    validation_loss=0
		    for i, (batch,labels) in enumerate(validation_loader):
		        with torch.no_grad():
		            y_h = cnn(batch.cuda())
		            validation_loss = loss_func(y_h,labels.cuda())
		    validation_losses.append(validation_loss)
		    #'''
		    print('Epoch:{}, training_loss:{}, validation_losses:{}'.format(e,training_loss,validation_loss))

		info = {'model':cnn,
		'optimizer':optimizer,
		'loss_func':loss_func,
		'epochs': epochs,
		'training_losses':training_losses,
		'validation_losses':validation_losses
		}

		import datetime,re
		name=re.findall('ConvNet[0-9]+',str(type(info['model'])))[0]+str(datetime.datetime.now())+'.pkl'
		f = open(name,'wb')
		pickle.dump(info,f)
		f.close()
		print('saved trained model in '+name)
	elif mode=='test':
		try:
			f = open(sys.argv[2],'rb')
		except:
			print("failled to read file, make sure you mentioned which file to test")
		info = pickle.load(f)
		print('Model restored')
		print(info)


		cnn=info['model']
		all_y = []
		all_y_h = []
		cnn.eval()
		with torch.no_grad():
		    for  i,(X,y) in enumerate(test_loader):
		        y_h = cnn(X.cuda()).cpu()
		        y_h = torch.argmax(F.softmax(y_h),dim=1)
		        if all_y==[]:
		            all_y=y[:]
		            all_y_h = y_h[:]
		            
		        else:
		            all_y = torch.hstack([all_y,y])
		            all_y_h = torch.hstack([all_y_h,y_h])

		print('Accuracy: ',(all_y==all_y_h).sum()/len(all_y))
		print('Precision: ',precision_score(all_y,all_y_h,average='weighted'))
		print('Recall: ',recall_score(all_y,all_y_h,average='weighted'))
		print('F1: ',f1_score(all_y,all_y_h,average='weighted'))
		print(confusion_matrix(all_y,all_y_h))