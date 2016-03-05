#ifndef CAFFE_GRAD_ORIENT_CONV_LAYER_HPP_
#define CAFFE_GRAD_ORIENT_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters, 
 *		  oriented along the dominant gradient at that location,
 *        and (optionally) adds biases. The output also gets cos and
 *		  sin of the gradient orientations appended to it.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class GradOrientConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size . The filter dimensions given by
   *  kernel_size for square filters .
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines. This GradOrient Covolution
   *	implementation only admits the CAFFE Engine unless some brave soul
   *	ports it to CUDNN after this funky layer has been proven to be useful.
   */
  explicit GradOrientConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  //Dunno if this can be mucked about with
  virtual inline const char* type() const { return "Convolution"; }
//  virtual inline const char* type() const { return "Gradient Oriented Convolution"; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //bottom[0] is the input to be convolved.
  //bottom[1] brings with it the gradient map (2 channels)
  virtual inline int MinBottomBlobs() const {return 2;}
  //There are two outputs, one being the convolution result and the other
  //being the gradient map of the same spatial dimensions as the input
  virtual inline int MinTopBlobs() const {return 1;}
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }


 protected:
  int pooled_height_, pooled_width_;

  Blob<double> gauss_kernel_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
 private:
  //This computes the gradient orientation using gaussian convolved input
  //gradient maps. The gaussian convolved gradient maps are output in top[1]
  Blob<Dtype> orientation_map_;
  //Construct this from the orientation map, based on whether linear or
  //or non-linear interpolation is being used.
  //Dimensions num_ x 4 x height x width  
  Blob<Dtype> intermediate_kernel_alphas_;
  //Number of rotated versions of the kernel. This would be a parameter
  //some day. Hard coded to 4
  int num_rotations_;
  //Rotated copies of the kernel. Storing these doesn't cost much
  //Filled in Reshape, before each forward pass. If caffe ever changes its
  //philosophy of calling Reshape everytime, move this to the forward pass
  //so that it contains updated copies of the weights.
  vector<Blob<Dtype>> intermediate_kernels_;
};

}  // namespace caffe

#endif  // CAFFE_GRAD_ORIENT_CONV_LAYER_HPP_
