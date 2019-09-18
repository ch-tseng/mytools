sudo apt-get update && sudo apt-get upgrade -y
sudo apt install net-tools -y
sudo apt autoremove -y
sudo apt-get purge wolfram-engine -y
sudo apt-get purge libreoffice* -y
sudo apt-get clean -y
sudo apt-get autoremove -y
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install build-essential cmake pkg-config -y
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install libjasper1 libjasper-dev -y
sudo apt-get install libgtk-3-dev -y
sudo apt-get install libatlas-base-dev gfortran -y
sudo apt-get install python3.6-dev -y
sudo apt-get install cmake -y
sudo apt-get install python3-dev -y
sudo apt-get install gcc g++ -y
sudo apt-get install gtk2-devel -y
sudo apt-get install libv4l-devel -y
sudo apt-get install ffmpeg-devel -y
sudo apt-get install gstreamer-plugins-base-devel -y
sudo apt-get install libpng-devel -y
sudo apt-get install libjpeg-turbo-devel -y
sudo apt-get install jasper-devel -y
sudo apt-get install openexr-devel -y
sudo apt-get install libtiff-devel -y
sudo apt-get install libwebp-devel -y


wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

sudo apt-get install virtualenv
mkdir ~/envs
cd ~/envs/
virtualenv AI -p python3
source ~/envs/AI/bin/activate
pip install numpy scipy

cd ~
wget https://github.com/opencv/opencv/archive/4.1.1.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.1.zip

unzip 4.1.1.zip
unzip opencv_contrib.zip
ln -s opencv-4.1.1 opencv
ln -s opencv_contrib-4.1.1 opencv_contrib

cd ~/opencv
mkdir build
cd build/

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=ON ..

make -j4
sudo make install
