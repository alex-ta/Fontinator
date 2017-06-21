How to Setup:


Install [anaconda](https://www.continuum.io/downloads)
Run commands:
1. conda create -n IDA python=3.6
2. activate IDA
3. conda install numpy pandas pillow matplotlib scipy scikit-learn theano h5py graphviz
4. pip install pydot-ng
5. conda install keras -c conda-forge
6. conda install opencv -c conda-forge
if you get an error message containing tensorflow you are using the wrong backend, check out to switch the backend

Tensorflow is currently not supported on 3.6
Easy way:
1. conda create --name tensorflow python=3.5
2. activate tensorflow
3. pip install tensorflow
* or
* pip install tensorflow-gpu
3. conda install numpy pandas pillow matplotlib scipy scikit-learn theano h5py 
4. conda install keras -c conda-forge
5. conda install opencv -c conda-forge
It also supports theano

To switch backends for keras:
1. Edit the jsonfile on your home directory (.keras) folder
2. The Json looks something like this:
  {
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_data_format": "channels_last"
  }
  change the backend to your prefered backend
You can also follow this [tutorial](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/).
