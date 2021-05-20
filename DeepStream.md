# Install Deepstream in the Jetson NX

#### DeepStream, as the name implies, is NVIDIA's platform for processing continuous frames through video or cameras.

    NVIDIA DeepStream simplifies the development of scalable intelligent video analytics (IVA) applications
    Developers can now use this to quickly build new applications to transform video into valuable insight.
    Applications for the DeepStream SDK include image classification, scene understanding, video categorization, content filtering etc.
### Pre-installed:
    
    Cuda, Cudnn must be installed before going to install deepstream. In the Jetson, the tensorrt is auto installed.

### Install Dependencies
``` python
$ sudo apt install \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1 
```
### Install librdkafka (to enable Kafka protocol adaptor for message broker)
``` python
$ git clone https://github.com/edenhill/librdkafka.git
``` 
### Config the library

``` python 
$ cd librdkafka
$ git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a
./configure
$ make
$ sudo make install
``` 
### Copy the generated libraries to the deepstream directory

``` 
$ sudo mkdir -p /opt/nvidia/deepstream/deepstream-5.1/lib
$ sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-5.1/lib
``` 
## Install the DeepStream SDK
    I preferred to use the tar file installation. 
1. Download the DeepStream 5.1 Jetson tar package deepstream_sdk_v5.1.0_jetson.tbz2, to the Jetson device.
    Go to this link and download deepstream
    <link> https://developer.nvidia.com/deepstream-getting-started </link>
2. Enter the following commands to extract and install the DeepStream SDK:
``` python
$  sudo tar -xvf deepstream_sdk_v5.1.0_jetson.tbz2 -C /
$ cd /opt/nvidia/deepstream/deepstream-5.1
$ sudo ./install.sh
$ sudo ldconfig
```

