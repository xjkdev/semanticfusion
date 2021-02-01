import caffe_pb2
import models
import torch
import numpy as np
from functools import reduce

caffemodel = 'inference.caffemodel'

model = caffe_pb2.NetParameter()
print('Loading caffemodel: ' + caffemodel)
with open(caffemodel, 'rb') as fp:
    model.ParseFromString(fp.read())


net = models.DeConvNet()
sd = net.state_dict()
print(sd.keys())
names = [l.name for l in model.layer]
print(names)


def product(iterable):
    return reduce(lambda a, b: a * b, iterable)


for l in model.layer:
    name = l.name.replace('-', '_')
    print(name)
    if name == 'deconv1_2_derelu1_2_0_split':
        continue
    if (name.startswith("conv") or name.startswith("deconv") or 
        name in ['fc6', 'fc7', 'fc6_deconv', 'class_score_nyu']):
        subkeys = ['.weight', '.bias']
        for i, subkey in enumerate(subkeys):
            curkey = name + subkey
            print(curkey, l.blobs[i].shape.dim, list(sd[curkey].shape))
            assert list(l.blobs[i].shape.dim) == list(sd[curkey].shape), \
                f"{l.name}, get {l.blobs[i].shape.dim}, expected {sd[curkey].shape}"
            tmpdata = l.blobs[i].data
            sd[curkey] = torch.Tensor(tmpdata).reshape(
                sd[curkey].shape)
    elif name.startswith("bn") or name.startswith("debn") or name in ['fc6_deconv_bn']:
        subkeys = ['.weight', '.bias']
        for i, subkey in enumerate(subkeys):
            curkey = name + subkey
            print(curkey, l.blobs[i].shape.dim, list(sd[curkey].shape))
            assert product(l.blobs[i].shape.dim) == product(sd[curkey].shape), \
                f"{l.name}, get {l.blobs[i].shape.dim}, expected {sd[curkey].shape}"
            sd[curkey] = torch.Tensor(
                l.blobs[i].data).reshape(
                sd[curkey].shape)
            
torch.save(sd, "tmp_sd.pt")
