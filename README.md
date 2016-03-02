###Bits and Bobs From Deepnet Experimentation

## New Layers With CPU and GPU Implementations
### L1 Loss Layer
https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/include/caffe/layers/l1_loss_layer.hpp

https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/src/caffe/layers/l1_loss_layer.cpp

https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/src/caffe/layers/l1_loss_layer.cu
#### Usage
Exactly like L2 Loss Layer

### SSIM Loss Layer (Structural Similarity)
https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/include/caffe/layers/ssim_loss_layer.hpp

https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/src/caffe/layers/ssim_loss_layer.cpp

https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/src/caffe/layers/ssim_loss_layer.cu
#### Parameters for SSIM
https://github.com/mehtadushy/DeepNetStuff/blob/master/caffe/src/caffe/proto/caffe.proto
#### Usage
```
layer {
  name: "mylosslayer"
  type: "SSIMLoss"
  bottom: "result"
  bottom: "ground_truth"
  top: "loss_vale"
  loss_weight: 1             # <- set whatever you fancy
  ssim_loss_param{
    kernel_size: 8           # <- The kernel size is linked to the gaussian variance (circular). The kernel encloses +/1 3*sigma 
    stride: 8                # <- Equal strides in both dimensions
    c1: 0.0001               # <- Let these be
    c2: 0.001                # <- Let these be
  }
}
```

