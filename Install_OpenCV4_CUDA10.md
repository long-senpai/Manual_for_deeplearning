# How to install OpenCV 4.1.0 with CUDA 10.0 in Ubuntu distro 18.04

First of all install update and upgrade your system:
    
        $ sudo apt update
        $ sudo apt upgrade
   
    
Then, install required libraries:

* Generic tools:

        $ sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall
    
* Image I/O libs
    ``` 
    $ sudo apt install libjpeg-dev libpng-dev libtiff-dev libjasper-dev
    ``` 
* Video/Audio Libs - FFMPEG, GSTREAMER, x264 and so on.
    ```
    $ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
    $ sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    $ sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
    $ sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
    ```
* OpenCore - Adaptive Multi Rate Narrow Band (AMRNB) and Wide Band (AMRWB) speech codec
    ```
    $ sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev
    ```
    
* Cameras programming interface libs
    ```
    $ sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
    $ cd /usr/include/linux
    $ sudo ln -s -f ../libv4l1-videodev.h videodev.h
    $ cd ~
    ```

* GTK lib for the graphical user functionalites coming from OpenCV highghui module 
    ```
    $ sudo apt-get install libgtk-3-dev
    ```
* Python libraries for python2 and python3:
    ```
    $ sudo apt-get install python3-dev python3-pip
    $ sudo -H pip3 install -U pip numpy
    $ sudo apt install python3-tesresources
    ```
* Parallelism library C++ for CPU
    ```
    $ sudo apt-get install libtbb-dev
    ```
* Optimization libraries for OpenCV
    ```
    $ sudo apt-get install libatlas-base-dev gfortran
    ```
* Optional libraries:
    ```
    $ sudo apt-get install libprotobuf-dev protobuf-compiler
    $ sudo apt-get install libgoogle-glog-dev libgflags-dev
    $ sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
    ```

We will now proceed with the installation (see the Qt flag that is disabled to do not have conflicts with Qt5.0).

    $ cd ~
    $ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip
    $ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
    $ unzip opencv.zip
    $ unzip opencv_contrib.zip
    
    $ echo "Create a virtual environtment for the python binding module"
    $ sudo pip install virtualenv virtualenvwrapper
    $ sudo rm -rf ~/.cache/pip
    $ echo "Edit ~/.bashrc"
    $ export WORKON_HOME=$HOME/.virtualenvs
    $ export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    $ source /usr/local/bin/virtualenvwrapper.sh
    $ mkvirtualenv cv -p python3
    $ pip install numpy
    
    $ echo "Procced with the installation"
    $ cd opencv-4.1.0
    $ mkdir build
    $ cd build
    
    $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D WITH_TBB=ON \
	-D WITH_CUDA=ON \
	-D BUILD_opencv_cudacodec=OFF \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D WITH_CUBLAS=1 \
	-D WITH_V4L=ON \
	-D WITH_QT=OFF \
	-D WITH_OPENGL=ON \
	-D WITH_GSTREAMER=ON \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_PC_FILE_NAME=opencv.pc \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.6/site-packages \
	-D OPENCV_EXTRA_MODULES_PATH=~/downloads/opencv/opencv_contrib-4.1.0/modules \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
	-D BUILD_EXAMPLES=ON ..
	

If you want to build the libraries statically you only have to include the *-D  BUILD_SHARED_LIBS=OFF*

    $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D WITH_TBB=ON -D WITH_CUDA=ON -D BUILD_opencv_cudacodec=OFF -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.6/site-packages -D OPENCV_EXTRA_MODULES_PATH=~/downloads/opencv/opencv_contrib-4.1.0/modules -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python -D BUILD_EXAMPLES=ON -D BUILD_SHARED_LIBS=OFF ..
    
In case you do not want to include include CUDA set *-D WITH_CUDA=OFF*     

    $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_opencv_cudacodec=OFF -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.6/site-packages -D OPENCV_EXTRA_MODULES_PATH=~/downloads/opencv/opencv_contrib-4.1.0/modules -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python -D BUILD_EXAMPLES=ON ..

Before the compilation you must check that CUDA has been enabled in the configuration summary printed on the screen.
```
--   NVIDIA CUDA:                   YES (ver 10.0, CUFFT CUBLAS NVCUVID FAST_MATH)
--     NVIDIA GPU arch:             30 35 37 50 52 60 61 70 75
--     NVIDIA PTX archs:

```

If it is fine proceed with the compilation (Use nproc to know the number of cpu cores):
    
    $ nproc
    $ make -j8
    $ sudo make install

Include the libs in your environment    
    
    $ sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
    $ sudo ldconfig
    
If you want to have available opencv python bindings in the system environment you should copy the created folder during the installation of OpenCV (* -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.6/site-packages *) into the *dist-packages* folder of the target python interpreter:

    $ sudo cp -r ~/.virtualenvs/cv/lib/python3.6/site-packages/cv2 /usr/local/lib/python3.6/dist-packages
    
    $ echo "Modify config-3.6.py to point to the target directory" 
    $ sudo nano /usr/local/lib/python3.6/dist-packages/cv2/config-3.6.py 
    
    ``` 
	    PYTHON_EXTENSIONS_PATHS = [
	    os.path.join('/usr/local/lib/python3.6/dist-packages/cv2', 'python-3.6')
	    ] + PYTHON_EXTENSIONS_PATHS
    ``` 

** TODO ADAPT EXAMPLE TO OPENCV 4.1 in C++ **
Verify the installation by compiling and executing the following example:
```
#include <iostream>
#include <ctime>
#include <cmath>
#include "bits/time.h"

//#include <opencv2/opencv.hpp>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <imgcodecs/imgcodecs.hpp>

#include <core/cuda.hpp>
#include <cudaarithm.hpp>
#include <cudaimgproc.hpp>

#define TestCUDA true

int main()
{
    std::clock_t begin = std::clock();

    try {
        cv::Mat srcHost = cv::imread("image.png");

        for(int i=0; i<1000; i++) {
            if(TestCUDA) {
                cv::cuda::GpuMat dst, src;
                src.upload(srcHost);

                //cv::cuda::threshold(src,dst,128.0,255.0, CV_THRESH_BINARY);
                cv::cuda::bilateralFilter(src,dst,3,1,1);

                cv::Mat resultHost;
                dst.download(resultHost);
            } else {
                cv::Mat dst;
                cv::bilateralFilter(srcHost,dst,3,1,1);
            }
        }

        //cv::imshow("Result",resultHost);
        //cv::waitKey();

    } catch(const cv::Exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }

    std::clock_t end = std::clock();
    std::cout << double(end-begin) / CLOCKS_PER_SEC  << std::endl;
}
```
Compile and execute:

    $ g++ `pkg-config opencv --cflags --libs` -o test test.cpp
    $ ./test

*If you have any problem try updating the nvidia drivers.*



### Source
- [pyimagesearch](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)
- [learnopencv](https://www.learnopencv.com/install-opencv-4-on-ubuntu-18-04/)
- [Tzu-cheng](https://chuangtc.com/ParallelComputing/OpenCV_Nvidia_CUDA_Setup.php)
- [Medium](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)
- [Previous Gist](https://gist.github.com/raulqf/a3caa97db3f8760af33266a1475d0e5e)
