sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev 
sudo apt-get install libx11-dev libgtk-3-dev
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install libboost-all-dev

wget http://dlib.net/files/dlib-19.17.zip
unzip dlib-19.17.zip
cd dlib-19.17
python setup.py install
