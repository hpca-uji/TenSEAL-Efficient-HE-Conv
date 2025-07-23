
<h1 align="center">
  TenSEAL library + efficient convolution methods
  <br>
</h1>

<h3 align="center">
  <br>
  This work extendend the TenSEAL library to perform homomorphic encryption convolutions on tensors
  <br>
</h3>



[TenSEAL](https://github.com/OpenMined/TenSEAL) is a library for doing homomorphic encryption operations on tensors, built on top of [Microsoft SEAL](https://github.com/Microsoft/SEAL). It provides ease of use through a Python API, while preserving efficiency by implementing most of its operations using C++. 
In this work, we have implemented a new matmul method, the im2row transform, a pooling function, and the direct convolution for the CKKSTensor class.

## Features

New Methods Added
mm_row and mm_row_:
Perform matrix multiplication between a CKKSTensor and a Plaintext. Both methods access data by rows to ensure coalesced memory access, improving performance. 
mm_row returns a new CKKSTensor.
mm_row_ performs the multiplication in-place, modifying the input tensor.

im2row and im2row_: Compute the im2row transformation of an encrypted matrix with shape (input_channels, h × w). The input is expected to already be in im2row format and these methods perform convolution via matrix-matrix multiplication.
These methods support concatenation of multiple convolutions approximated using this approach.
Parameters include kernel size, stride, input channels, padding, and output channels.
The kernel weights have shape (output_channels, input_channels × kernel_height × kernel_width).

conv_direct and conv_direct_:
Perform convolution directly using the im2row strategy but without explicitly constructing the im2row matrix, thus avoiding unnecessary zero multiplications caused by padding.
Parameters include kernel size, stride, input channels, padding, output channels, weights, and bias.
Kernel weights shape: (output_channels, input_channels × kernel_height × kernel_width), matching the flattened im2row format used for matrix multiplication.

pooling_layer and pooling_layer_: Implement the pooling approach described in the [Cryptonets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) paper.


## Installation


If you want to install TenSEAL-Efficient-HE-Conv from the repository, you should first make sure to have the requirements for your platform (listed above) and [CMake (3.14 or higher)](https://cmake.org/install/) installed.

```bash
$ git clone http://lorca.act.uji.es/gitlab/deep-learning/tenseal_ckks_convolution.git
```


## Publications

Núria Moreno-Chamorro, Maribel Castillo, José I. Aliaga, Manuel F. Dolz 
Universitat Jaume I, Castellón de la Plana, Spain
Emails: {morenon, castillo, aliaga, dolzm}@uji.es 
"Analyzing Performance–Memory–Security Trade-Offs of Convolutions for DNN Inference on Homomorphically Encrypted Data" 
ISPDC 2025




