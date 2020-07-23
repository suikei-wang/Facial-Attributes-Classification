# test a certain example using the best model
import torchvision.transforms as transforms
import torch
from PIL import Image


def main():
	transform = transforms.Compose(
	        [transforms.Resize(224),
	         transforms.ToTensor(),
	         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	         
	image_set = Image.open('/content/test.jpg')

	image_tensor = transform(image_set)
	image = torch.unsqueeze(image_tensor, 0)
	image = image.to('cuda')
	output = model(image)
	result = output > 0.5
	result = result.cpu().numpy()


	for t in range(len(attributes)):
	    if result[0][t] == True:
	       print("Attribute: \033[1;35m%s \033[0m, \033[1;35m%s \033[0m" % (attributes[t], result[0][t]))
	    else:
	       print("Attribute: %s, %s" % (attributes[t], result[0][t]))


if __name__ == "__main__":
    main()