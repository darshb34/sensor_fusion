import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

def my_forward(images):
    resnet50 = models.resnet50(pretrained=True).cuda()
    modules=list(resnet50.children())[:-5]
    resnet50=nn.Sequential(*modules)
    # for p in resnet50.parameters():
    #     p.requires_grad = False
        
    # img = torch.Tensor(2,3, 640, 526).normal_() # random image
    # image_batch_cur = np.expand_dims(numpy_image, axis=0)

    # img_var = Variable(images) # assign it to a variable
    features_var = resnet50(images) # get the output from the last hidden layer of the pretrained resnet
    # features = features_var.data # get the tensor out of the variable
    return features_var
    # print(features.shape)