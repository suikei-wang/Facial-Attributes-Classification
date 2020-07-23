from model import MobileNet
import torch
from torch import nn
from matplotlib import pyplot as plt

def main():
	# define empty list to store the losses and accuracy for ploting
	train_all_losses2 = []
	train_all_acc2 = []
	val_all_losses2 = []
	val_all_acc2 = []
	test_all_losses2 = 0.0
	# define the training epoches
	epochs = 100

	# instantiate Net class
	mobilenet = MobileNet()
	# use cuda to train the network
	mobilenet.to('cuda')
	#loss function and optimizer
	criterion = nn.BCELoss()
	learning_rate = 1e-3
	optimizer = torch.optim.Adam(mobilenet.parameters(), lr=learning_rate, betas=(0.9, 0.999))

	%load_ext memory_profiler

	best_acc = 0.0

	for epoch in range(epochs):
	    train(mobilenet, epoch, train_all_losses2, train_all_acc2)
	    acc = validation(mobilenet, val_all_losses2, val_all_acc2, best_acc)
	    # record the best model
	    if acc > best_acc:
	      checkpoint_path = './model_checkpoint.pth'
	      best_acc = acc
	      # save the model and optimizer
	      torch.save({'model_state_dict': mobilenet.state_dict(),
	              'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
	      print('new best model saved')
	    print("========================================================================")

	checkpoint_path = './model_checkpoint.pth'
	model = MobileNet().to('cuda')
	checkpoint = torch.load(checkpoint_path)
	print("model load successfully.")

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	model.eval()
	attr_acc = []
	test(model, attr_acc=attr_acc)

	# plot results
	plt.figure(figsize=(8, 10))
	plt.barh(range(40), [100 * acc for acc in attr_acc], tick_label = attributes, fc = 'brown')
	plt.show()

	plt.figure(figsize=(8, 6))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss')
	plt.grid(True, linestyle='-.')
	plt.plot(train_all_losses2, c='salmon', label = 'Training Loss')
	plt.plot(val_all_losses2, c='brown', label = 'Validation Loss')
	plt.legend(fontsize='12', loc='upper right')
	plt.show()

	plt.figure(figsize=(8, 6))
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Accuracy')
	plt.grid(True, linestyle='-.')
	plt.plot(train_all_acc2, c='salmon', label = 'Training Accuracy')
	plt.plot(val_all_acc2, c='brown', label = 'Validation Accuracy')
	plt.legend(fontsize='12', loc='lower right')
	plt.show()

if __name__ == "__main__":
    main()











