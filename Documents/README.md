How to Setup:


Install anaconda (https://www.continuum.io/downloads)
Run commands:
>> conda create -n IDA python=3.6
>> activate IDA
>> conda install numpy pandas pillow matplotlib scipy scikit-learn theano h5py
>> conda install keras -c conda-forgHG
>> conda install opencv -c conda-forge


if you get an error message containing tensorflow you are using the wrong backend,
follow step three in the linked [tutorial](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/).
