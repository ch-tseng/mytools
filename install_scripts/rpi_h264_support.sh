cd ~
git clone --depth 1 http://git.videolan.org/git/x264
cd x264
./configure --host=arm-unknown-linux-gnueabi --enable-static --disable-opencl
make -j4
sudo make install
cd ~
git clone git://source.ffmpeg.org/ffmpeg --depth=1
cd ffmpeg
./configure --arch=armel --target-os=linux --enable-gpl --enable-libx264 --enable-nonfree
make -j4
sudo make install
