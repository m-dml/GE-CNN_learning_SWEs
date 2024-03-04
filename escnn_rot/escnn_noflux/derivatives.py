import torch
import math
from examples.deep_model import get_param

params = get_param.params()

def toCuda(x):
	if type(x) is tuple:
		return [xi.cuda() if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x

def toCpu(x):
	if type(x) is tuple:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()



def vector2HSV(vector,plot_sqrt=False):
	"""
	transform vector field into hsv color wheel
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = torch.ones(values.shape).cuda()
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()
