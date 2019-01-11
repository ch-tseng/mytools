sudo apt-get update
sudo apt-get upgrade –y
sudo apt install python-pip python3-pip
sudo apt-get install -y libusb-1.0-0-dev libprotobuf-dev
sudo apt-get install -y libleveldb-dev libsnappy-dev
sudo apt-get install -y libopencv-dev
sudo apt-get install -y libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y libatlas-base-dev git automake 
sudo apt-get install -y byacc lsb-release cmake 
sudo apt-get install -y libgflags-dev libgoogle-glog-dev 
sudo apt-get install -y liblmdb-dev swig3.0 graphviz 
sudo apt-get install -y libxslt-dev libxml2-dev 
sudo apt-get install -y gfortran 
sudo apt-get install -y python3-dev python-pip python3-pip 
sudo apt-get install -y python3-setuptools python3-markdown 
sudo apt-get install -y python3-pillow python3-yaml python3-pygraphviz
sudo apt-get install -y python3-h5py python3-nose python3-lxml 
sudo apt-get install -y python3-matplotlib python3-numpy 
sudo apt-get install -y python3-protobuf python3-dateutil 
sudo apt-get install -y python3-skimage python3-scipy 
sudo apt-get install -y python3-six python3-networkx
sudo pip3 install opencv_contrib_python
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libjasper-dev
sudo apt-get install -y libqtgui4
sudo apt-get install -y libqt4-test
sudo apt-get install -y python3-pyqt5
sudo pip3 install imutils
sudo apt install -y cython
sudo -H pip3 install cython
sudo -H pip3 install numpy
sudo -H pip3 install pillow
sudo pip3 uninstall tensorflow
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.7.0/tensorflow-1.7.0-cp35-none-linux_armv7l.whl
sudo pip3 install tensorflow-1.7.0-cp35-none-linux_armv7l.whl
rm tensorflow-1.7.0-cp35-none-linux_armv7l.whl
cd ~; git clone -b ncsdk2 http://github.com/Movidius/ncsdk
cd ncsdk/
sudo make install
