#include <vector>

#include "caffe/layers/grad_orient_conv_layer.hpp"

namespace caffe {

//Blobs are N x C x H x W and axes/indices are in that order
//          0   1   2   3

template <typename Dtype>
void GradOrientConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //Stuff handled here:
  //- Initialize kernel size, pooling and pad sizes
  //- Check output channels and group mismatch
  //- Setup learned kernels
  //- Initialize gaussian kernel
  //- Setup intermediate_kernels_

  //Parameterize in the future
  num_rotations_ = 4;
  
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  CHECK_EQ(num_spatial_axes_, 2)
        << "GradOrient only supports 2D Convolutions. I can't wrap my head around"
		<< "how 3D convolutions could be made rotational invariant in this way.";
  //Complicated way of initializing a Blob to store the spatial dimensions
  //of the kernel
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 2));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  //Also ensure that the kernels are square
  CHECK_EQ(kernel_shape_data[0], kernel_shape_data[1])
        << "The Kernels should be square.";

  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";

  //Reverse_dimensions is false, so conv_out_channels is num_output_
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
	//Setting up intermediate kernels
	for (int i = 0; i < num_rotations_; ++i) {
		//Acceptable here because the spatial span along x and y is the same.
		//If changing the kernels to rectangular kernels, resize these
		//accordingly
		intermediate_kernels_.pushback(*blob_[0]);
	}
  }

  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  //Create the gaussian kernel
  int kernel_w, kernel_h;
  kernel_h = kernel_shape_.cpu_data()[0];
  kernel_w = kernel_shape_.cpu_data()[1];
  gauss_kernel_.Reshape(1,1,kernel_h,kernel_w);
  double* gaussian = gauss_kernel_.mutable_cpu_data();
  double sigma = (kernel_w+kernel_h)/Dtype(12);
  double gauss_sum = 0;
  for (int h = 0; h < kernel_h; ++h) {
      for (int w = 0; w < kernel_w; ++w) {
	  gaussian[h * kernel_w_ + w] = 
		  exp(-(pow((h - kernel_h/2.0),2) + pow((w - kernel_w/2.0),2)) / (2.0* sigma * sigma))
		      / (2 * 3.14159 * sigma * sigma);
	   gauss_sum += gaussian[h* kernel_w_ + w];
      }
  }
  caffe_scal(gauss_kernel_.count(), 1.0/gauss_sum, gaussian);
}

template <typename Dtype>
void GradOrientConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //Stuff handled here:
  //- Check if bottom spatial sizes and num images are the same
  //- Check if bottom[1] has 2 channels
  //- Initialize top[0], top[1] and top[2]
  //- Initialize intermediate kernels
  //- Initialize orientation map
  //- Initialize intermediate_kernel_alphas from the orientation map
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))
      << "both bottoms ought to have the same height.";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))
      << "both bottoms ought to have the same width.";
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "both bottoms ought to have the same num.";
  CHECK_EQ(bottom[1]->shape(1), 2)
      << "gradient map would have 2 channels, Gx and Gy.";

  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  top[0]->Reshape(top_shape);
  top_shape[1] = 2;
  top[1]->Reshape(top_shape);
  top[2]->Reshape(top_shape);
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  //orientation map would be top sized in n, h and w
  top_shape[1] = 1;
  orientation_map_.resize(top_shape);
  intermediate_kernel_alphas_.resize(alpha_shape);
  vector<int> alpha_shape(4);
  //Alpha maps would be top sized in n, h and w
  alpha_shape[0] = top_shape[0];
  alpha_shape[1] = 4;
  alpha_shape[2] = top_shape[2];
  alpha_shape[3] = top_shape[3];
  intermediate_kernel_alphas_.resize(alpha_shape);
  //Create num_rotations_ copies of the weight kernel in intermediate_kernels_
  //In the future this would be done through a function call that takes
  //num_rotations as input and create as many interpolated/rotated copies of the 
  //kernels.
  Dtype* kernels = blobs_[0].cpu_data();
  vector<Dtype*> rot_kernel;
  for(int i = 0; i < num_rotations_; ++i){
     rot_kernel.pushback(intermediate_kernels_[i].mutable_cpu_data());
  }
	 
  //Rotating counter clockwise
  for(int n = 0; n < blobs_[0]->shape(0); ++n){
	for(int c = 0; c < blobs_[0]->shape(1); ++c){
	 for(int h = 0; h < blobs_[0]->shape(2); ++h){
	  for(int w = 0; w < blobs_[0]->shape(3); ++w){
		 int q = h* blobs_[0]->shape(3) + w; 
		 int r = h* blobs_[0]->shape(3) + w; 
	     rot_kernel[0][q] = kernels[r];

		 r = w * blobs_[0]->shape(3) +blobs_[0]->shape(2)- h; 
	     rot_kernel[1][q] = kernels[r];

		 r = (blobs_[0]->shape(2)- h) * blobs_[0]->shape(3) +blobs_[0]->shape(3)- w;
	     rot_kernel[2][q] = kernels[r];

		 r = (blobs_[0]->shape(3)-w) * blobs_[0]->shape(3) + h; 
	     rot_kernel[3][q] = kernels[r];
	  }
	 }

	  for(int i = 0; i < num_rotations_; ++i){
		 rot_kernel[i] += blobs_[0]->offset(0,1);
	  }
	}//c
  }//n


}
template <typename Dtype>
void GradOrientConvolutionLayer<Dtype>::GaussConvolveHelper(const Blob<Dtype>& in, Blob<Dtype>& out){
    int N = in.shape(0);
	//int C = in.shape(1);
	int H = in.shape(2);
	int W = in.shape(3);
	int PH = out.shape(2);
	int PW = out.shape(3);

	int stride_h =stride_data[0]; 
	int stride_w =stride_data[1]; 

	int pad_h =pad_data[0]; 
	int pad_w =pad_data[1]; 
	
	int kernel_h = kernel_shape_data[0]; 
	int kernel_w = kernel_shape_data[1]; 

    const Dtype* in_data = in.cpu_data();    
    Dtype* out_data = out.mutable_cpu_data();
    caffe_set(out.count(), Dtype(0), out_data);
    const double* gaussian = gauss_kernel_.cpu_data();

    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < PH; ++ph) {
          for (int pw = 0; pw < PW; ++pw) {
            int hstart = ph * stride_h  - pad_h;
            int wstart = pw * stride_w  - pad_w;
            int hend = min(hstart + kernel_h, H );
            int wend = min(wstart + kernel_w, W );
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                out_data[ph * PW + pw] +=
                    gaussian[(h-hstart)*kernel_w+(w-wstart)]*in_data[h * W + w];
              }
            }
	  }
        }
        in_data += in.offset(0, 1);
        out_data += out.offset(0, 1);
      }
    }
}

template <typename Dtype>
void GradOrientConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
  pooled_width_ = output_shape_[1];
  pooled_height_ = output_shape_[0];
}

template <typename Dtype>
void GradOrientConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
  //Gauss convolve bottom[1] and create top[2]
  GaussConvolveHelper(*bottom[1], *top[2]);

  //Use top[2] to create the orientation map
  const Dtype* top2_x = top[2]->cpu_data();
  const Dtype* top2_y = top[2]->cpu_data() + top[2]->offset(0,1);
  Dtype* orientation = orientation_map_.mutable_cpu_data();
  int top_spatial_size = top[2]->count(2);
  for(int n = 0; n < top[2]->shape(0); ++n){
    caffe_atan2(top_spatial_size,top2_y, top2_x, orientation);  
	orientation += orientation_map_.offset(1);
	top2_x += top[2]->offset(1);
	top2_y += top[2]->offset(1);
  }

  //Use the orientation map to create top[1]. 
  Dtype* top1_0 = top[1]->mutable_cpu_data();
  Dtype* top1_1 = top[1]->mutable_cpu_data() + top[1]->offset(0,1);
  const Dtype* orientation = orientation_map_.cpu_data();
  int top_spatial_size = top[1]->count(2);
  for(int n = 0; n < top[2]->shape(0); ++n){
    caffe_sin(top_spatial_size, orientation, top1_0);  
    caffe_cos(top_spatial_size, orientation, top1_1);  
	orientation += orientation_map_.offset(1);
	top1_0 += top[1]->offset(1);
	top1_1 += top[1]->offset(1);
  }

  //Fill up intermediate_kernel_alphas_ based on orientation map
  const Dtype* orient = orientation_map_->cpu_data();
  Dtype* alphas = intermediate_kernel_alphas_.mutable_cpu_data();

  for(int n = 0; n < orientation_map_.shape(0); ++n){
	//for(int c = 0; c < bottom[1]->shape(1); ++c){ //1 channel in orientation
	 for(int h = 0; h <orientation_map_.shape(2); ++h){
	  for(int w = 0; w < orientation_map_.shape(3); ++w){
		  //Spatial Location in orientation map 
		 int q = h* orientation_map_.shape(3) + w; 
		 Dtype angle = 180. * orient[q] / M_PI; 
		 //Spatial Location in intermediate_kernel_alphas_
		 int r0 = ((0*orientation_map_.shape(2))+h)*orientation_map_.shape(3) + w;
		 int r1 = ((1*orientation_map_.shape(2))+h)*orientation_map_.shape(3) + w;
		 int r2 = ((2*orientation_map_.shape(2))+h)*orientation_map_.shape(3) + w;
		 int r3 = ((3*orientation_map_.shape(2))+h)*orientation_map_.shape(3) + w;
		 if((angle>=-180) && (angle<-90)){
			 // between 2 and 3
			 Dtype wght = (angle+180)/90; //0 to 1
			 alphas[r2] =  1-wght;
			 alphas[r3] =  wght;
		 }
		 else if((angle>=-90) && (angle<0)){
			 // between 3 and 0
			 Dtype wght = (angle+90)/90; //0 to 1
			 alphas[r3] =  1-wght;
			 alphas[r0] =  wght;
		 }
		 else if((angle>=0) && (angle<90)){
			 // between 0 and 1
			 Dtype wght = (angle)/90; //0 to 1
			 alphas[r0] =  1-wght;
			 alphas[r1] =  wght;
		 }
		 else if((angle>=90) && (angle<=180)){
			 // between 1 and 2
			 Dtype wght = (angle-90)/90; //0 to 1
			 alphas[r1] =  1-wght;
			 alphas[r2] =  wght;
		 }
	  }
	 }
	 orient+=orientation_map_->offset(1);
	 alphas+=intermediate_kernel_alphas_.offset(1);
	//}//c
  }//n

  //Do the convolutions and construct top[0]
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  //Make a temp blob to cache the resulting convolutions of the kernels
  vector<int> out_shape;
  out_shape.push_back(1);
  out_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    out_shape.push_back(output_shape_[i]);
  }
  Blob<Dtype> output_shaped_blob;
  output_shaped_blob.resize(out_shape);
  Dtype* temp_output = output_shaped_blob.mutable_cpu_data();

    Dtype* top_data = top[i]->mutable_cpu_data();
	
  for (int n = 0; n < this->num_; ++n) {
	vector<Blob<Dtype>> intermediate_results;
	for(int i = 0; i < num_rotation_; ++i){
		const Dtype* weight = intermediate_kernels_[i].cpu_data();
		this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
         temp_output);
		//Multiply with the correct spatial weight before storing
		//it in intermediate_results so that later they can be put
		//into top more easily.
		const Dtype* temp = output_shaped_blob.cpu_data();
		const Dtype* alphas = intermediate_kernel_alphas_.cpu_data();

		  for(int c = 0; c < num_output_ ; ++c){
		      alphas = alphas +  
				  temp indexed by channel
			  caffe_mul( spatial ,temp,alphas, temp_output)  
		  }
		  intermediate_results.pushback(output_shaped_blob);
		}
		//Sum up all four and put in top
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void GradOrientConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->t op_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradOrientConvolutionLayer);
#endif

INSTANTIATE_CLASS(GradOrientConvolutionLayer);

}  // namespace caffe
