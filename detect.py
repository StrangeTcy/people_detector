from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import argparse


#some constants and useful definitions
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
classes = utils.load_classes(class_path)

# standard yolo_v3 values
img_size=416
conf_thres=0.8
nms_thres=0.4

Tensor = torch.cuda.FloatTensor


# this might be an overly long and complicated way to do it
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')   



def prepare_image(image):

	# check whether image exists
	if not os.path.exists(image):
		print ("Sorry, this file doesn't seem to exist")
	else:
		img = Image.open(image)
	
	# scale and pad image
	ratio = min(img_size/img.size[0], img_size/img.size[1])
	imw = round(img.size[0] * ratio)
	imh = round(img.size[1] * ratio)
	img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])

	# convert image to Tensor
	image_tensor = img_transforms(img).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input_img = Variable(image_tensor.type(Tensor))

	return input_img


def get_model(config_path = config_path, 
			  weights_path = weights_path, 
			  class_path = class_path, 
			  img_size = img_size, 
			  *cuda):
	# Load model and weights
	# print ("Loading model")
	model = Darknet(config_path, img_size=img_size)
	model.load_weights(weights_path)
	if cuda:
		if torch.cuda.is_available():
			model.cuda()
	model.eval()
	
	return model


def detect_image(det_model, img, req_class="person"):
	# run inference on the model and get detections
	with torch.no_grad():
	    detections = det_model(img)
	    detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
	
	classes = utils.load_classes(class_path)

	# get all the unique detected classes' names 
	# (get actual string names, put them in a list, 
	# ensure uniqueness by converting list to set)
	det_classes = set([classes[int(cls_pred)] for cls_pred in detections[0][:,-1]])
	
	if req_class in det_classes:
		print ("Ok, this image seems to have people")
		return detections[0]
	else:
		print ("It seems that no people are detected in this image. Proceed?")
		return None
	


def show_image(img_path, detections, classes=classes):
	# Get bounding-box colors
	cmap = plt.get_cmap('tab20b')
	colors = [cmap(i) for i in np.linspace(0, 1, 20)]

	img = np.array(Image.open(img_path))
	# plt.figure()
	fig, ax = plt.subplots(1, figsize=(12,9))
	ax.imshow(img)

	pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
	pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
	unpad_h = img_size - pad_y
	unpad_w = img_size - pad_x


	unique_labels = detections[:, -1].cpu().unique()
	n_cls_preds = len(unique_labels)
	bbox_colors = random.sample(colors, n_cls_preds)
    
    # browse detections and draw bounding boxes
	for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
	    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
	    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
	    y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
	    x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
	    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
	    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
	    ax.add_patch(bbox)
	    plt.text(x1, y1, 
	    	    s=classes[int(cls_pred)], 
	    	    color='white', 
	    	    verticalalignment='top',
	            bbox={'color': color, 'pad': 0})

	plt.axis('off')
	
	# save image
	print ("Saving image")

	# Ok, this replacement can handle any file format(.jpeg, .png, .bmp ...)
	# but may cause trouble if there's a '.' in the filename 
	plt.savefig(img_path.replace(".", "-det."), bbox_inches='tight', pad_inches=0.0)
	
	plt.show() 


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str,
                        help="name of an image to process")
parser.add_argument("--cuda", type=str2bool,nargs='?',
                        const=True,default=True,
                        help="whether to use CUDA")
parser.add_argument("--time", type=str2bool,nargs='?',
                        const=True,default=False,
                        help="show inference time")
                      

def main(args):
	img_path = args.image
	inp_img = prepare_image(img_path)
	model = get_model(config_path, 
					  weights_path, 
		              class_path, 
		              img_size,
		              args.cuda)
	
	if args.time:
		prev_time = time.time()
	
	# get detections
	req_class = "person" # only detect people in an image
	detections = detect_image(model, inp_img, req_class)
	
	if args.time:
		inference_time = datetime.timedelta(seconds=time.time() - prev_time)
		print ("Inference Time: {}".format(inference_time))

	if detections is not None:
		show_image(img_path, detections, classes)
	

	 
        
if __name__ == '__main__':
	# let's get all our arguments
	args = parser.parse_args() 
	main(args)