
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
In this work, we have implemented the im2row transform, a pooling function, and the direct convolution for the CKKSTensor class.

## Features

The new methods added are: 

- mm_row and mm_row_: Matrix- multiplication of a CKKSTensor and Plaintext. 
  Both are accesed by rows to ensure coalesced memory access and improving performance.
  The mm_row method returns a new CKKSTensor, while mm_row_ performs the transformation in-place.

- im2row and im2row_: Compute the im2row transform of an encrypted matrix of shape (input_channels, h*w). 
  The input shape is the one obtained after apply the im2row transform and perform the convolution via matrix-matrix multiplication.
  This method supports the concatenation of different convolutions that have been approximated using this approach. 

- conv_direct and conv_direct_: Performs convolution directly following the im2row strategy, while skipping the explicit construction of the im2row matrix. 
  Efficiently avoids zero-product computations caused by padding.

- pooling_layer and pooling_layer_: Peform the pooling aproach of the [Cryptonets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) network.



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




