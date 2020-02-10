sudo apt-get update
sudo apt-get purge wolfram-engine -y
sudo apt-get purge libreoffice*  -y
sudo apt-get clean
sudo apt-get autoremove -y
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install build-essential cmake unzip pkg-config -y
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
sudo apt-get install libgtk-3-dev -y
sudo apt-get install libcanberra-gtk* -y
sudo apt-get install libatlas-base-dev gfortran -y
sudo apt-get install python3-dev -y
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/get-pip.py ~/.cache/pip
cd ~
mkdir openvino

virtualenv -p python3 envAI
source ~/envAI/bin/activate
echo "source ~/envAI/bin/activate" >> ~/.bashrc
pip install numpy
pip install "picamera[array]"
pip install imutils

cd ~/envAI/lib/python3.5/site-packages/
ln -s ~/openvino/inference_engine_vpu_arm/python/python3.5/cv2.cpython-35m-arm-linux-gnueabihf.so
echo "source ~/openvino/inference_engine_vpu_arm/bin/setupvars.sh" >> ~/.bashrc


cd openvino
wget http://download.01.org/openvinotoolkit/2018_R5/packages/l_openvino_toolkit_ie_p_2018.5.445.tgz
tar -zxvf l_openvino_toolkit_ie_p_2018.5.445.tgz

