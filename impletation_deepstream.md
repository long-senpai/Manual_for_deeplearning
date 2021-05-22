# TUTORIAL GUIDE TO CONVERT THE YOLOV4 DARKNET TO TENSORRT AND RUN DEEPSTREAM

> Pre-installed and Computer information:
> Ubuntu 20.04, GTX 1650 4 GB, 16 core
> Cuda 11.1, CuDNN 8., TensorRT 7.2.3.4 
> Anaconda, Pytorch installed, ... 

### Link Reference:
[Here](https://spyjetson.blogspot.com/2020/08/xavier-nx-deepstream-50-5-running.html)
    
- Make sure that you have installed the deepstream SDK, TensorRT on you computer.

## Clone the source code:
``` 
$ https://github.com/long-senpai/pytorch-YOLOv4.git
$ cd pytorch-YOLOv4
$ pip3 install onnxruntime
```
- If you have the Conda environment, activate it. 
## Convert DarkNet Model to ONNX Model

- Make sure that you have the Yolov4 weight file, cfg file, data, names.
- In order to convert the Darknet YOlov4 model to the onnx.
- Demo_darknet2onnx in the phytorch-YOLOV4 directory.Convert to ONNX model using py file. This command creates a new **yolov4_1_3_608_608_static.onnx** file. - 608_608 are the input shape of model, depend on you config file. 
``` 
$ python3 demo_darknet2onnx.py ./cfg/yolov4.cfg yolov4.weights ./data/giraffe.jpg 1
``` 
- the "./data/giraffe.jpg" just is a image for testing the ONNX model after converted. 

## Convert ONNX Model to TensorRT Model
- Depend on the way you used to instal the TensoRT, if you had isntalled by DEB package, the TensorRT location should same as below, if you installed by Tar file. The TensorRT folder may look like:
- /home/kikai/Downloads/TensorRT-7.2.3.4/bin/trtexec
```
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov4_1_3_608_608_static.onnx --explicitBatch --saveEngine=yolov4_1_3_608_608_fp16.engine --workspace=4096 --fp16
```
- One more parameter we should to concern about is the --Worksapce=4096
- If your computer have more than 4 GB of GPU, you can use that. Otherwise, if the GPU = 2 GB you have to change that parameter to 2048 instead. 
- The progress to convert ONNX to TensoRT model take a lot of times. 
- Once you finish this task a new model .engine is created. "yolov4_1_3_608_608_fp16.engine"


1. Copy the "nvdsparsebbox_Yolo.cpp" file to the folder of deestream. You may want to backup the existed file on that folder.
``` 
$ cd /opt/nvidia/deepstream/deepstream/sources/objectDetector_Yolo/nvdsinfer_custom_impl_Yolo
```
2. Open the **Make** file in /nvdsinfer_custom_impl_Yolo and define the Cuda Version... 
``` cmake 
CUDA_VER?=11.1
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
``` 
Or you can run the follows command. 
```
$ export CUDA_VER=10.2
$ sudo make clean
$ sudo make 
$ ls -al
```
* Then you will see a library .so file just be created.
* EX: "libnvdsinfer_custom_impl_Yolo.so"

## Config the Deepstream app.

- On the previous steps, you had download the Pytorch-yoloV4 source, then go to the to deepstream folder you will see a list file:
    - config_infer_primary_yoloV4.txt
    - deepstream_app_config_yoloV4.txt
    - labels.txt
    - nvdsinfer_custom_impl_Yolo/
    - Readme.md 
```
$ cd pytorch-YOLOv4/DeepStream
$ pwd 
```
- Open the config_infer_primary_yoloV4.txt file and change some parameter, for example: 
 ``` md  
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
#0=RGB, 1=BGR
model-color-format=0
#model-engine-file=/home/kikai/deep_stream/pytorch-YOLOv4/yolov4_1_3_608_608_fp16.engine
#labelfile-path=/home/kikai/deep_stream/pytorch-YOLOv4/DeepStream/labels.txt
model-engine-file=/home/kikai/deep_stream/pytorch-YOLOv4/yolov4_1_3_512_512_fp16.engine
labelfile-path=/home/kikai/deep_stream/pytorch-YOLOv4/yoloscaled/label.txt
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode=2
num-detected-classes=80
gie-unique-id=1
network-type=0
is-classifier=0
## 0=Group Rectangles, 1=DBSCAN, 2=NMS, 3= DBSCAN+NMS Hybrid, 4 = None(No clustering)
cluster-mode=4
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseCustomYoloV4
custom-lib-path=/opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet
#scaling-filter=0
#scaling-compute-hw=0
#output-blob-names=2012

[class-attrs-all]
nms-iou-threshold=0.6
pre-cluster-threshold=0.4
``` 
### And next is the deepstream_app_config_yoloV4.txt file
``` md
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5
#gie-kitti-output-dir=streamscl

[tiled-display]
enable=0
rows=1
columns=1
width=1280
height=720
gpu-id=0
#(0): nvbuf-mem-default - Default memory allocated, specific to particular platform
#(1): nvbuf-mem-cuda-pinned - Allocate Pinned/Host cuda memory, applicable for Tesla
#(2): nvbuf-mem-cuda-device - Allocate Device cuda memory, applicable for Tesla
#(3): nvbuf-mem-cuda-unified - Allocate Unified cuda memory, applicable for Tesla
#(4): nvbuf-mem-surface-array - Allocate Surface Array memory, applicable for Jetson
nvbuf-memory-type=0

[source0]
enable=1
#Type - 1=CameraV4L2 2=URI 3=MultiURI
type=3
uri=file:/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_1080p_h264.mp4
#uri=file:/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264
num-sources=1
gpu-id=0
# (0): memtype_device   - Memory type Device
# (1): memtype_pinned   - Memory type Host Pinned
# (2): memtype_unified  - Memory type Unified
cudadec-memtype=0

#For Screen Output
[sink0]
enable=1
#Type - 1=FakeSink 2=EglSink 3=File
type=2
sync=0
source-id=0
gpu-id=0
nvbuf-memory-type=0

#For File Output
[sink1]
enable=1
#Type - 1=FakeSink 2=EglSink 3=File
type=3
sync=0
source-id=0
gpu-id=0
nvbuf-memory-type=0
#1=mp4 2=mkv
container=1
#1=h264 2=h265
codec=1
output-file=yolov4.mp4

[osd]
enable=1
gpu-id=0
border-width=1
text-size=12
text-color=1;1;1;1;
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=0
clock-x-offset=800
clock-y-offset=820
clock-text-size=12
clock-color=1;0;0;0
nvbuf-memory-type=0

[streammux]
gpu-id=0
##Boolean property to inform muxer that sources are live
live-source=0
batch-size=1
##time out in usec, to wait after the first buffer is available
##to push the batch even if the complete batch is not formed
batched-push-timeout=40000
## Set muxer output width and height
width=1280
height=720
##Enable to maintain aspect ratio wrt source, and allow black borders, works
##along with width, height properties
enable-padding=0
nvbuf-memory-type=0

# config-file property is mandatory for any gie section.
# Other properties are optional and if set will override the properties set in
# the infer config file.
[primary-gie]
enable=1
gpu-id=0
model-engine-file=/home/spypiggy/src/pytorch-YOLOv4/yolov4_1_3_608_608_fp16.engine
labelfile-path=labels.txt
#batch-size=1
#Required by the app for OSD, not a plugin property
bbox-border-color0=1;0;0;1
bbox-border-color1=0;1;1;1
bbox-border-color2=0;0;1;1
bbox-border-color3=0;1;0;1
interval=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=/home/spypiggy/src/pytorch-YOLOv4/DeepStream/config_infer_primary_yoloV4.txt

[tracker]
enable=0
tracker-width=512
tracker-height=320
ll-lib-file=/opt/nvidia/deepstream/deepstream-5.0/lib/libnvds_mot_klt.so

[tests]
file-loop=0
```

# TO RUN

``` md 
$ cd /src/pytorch-YOLOv4
$ deepstream-app -c ./Deepstream/deepStream_app_config_yoloV4.txt
```

#### **Tips** : If you look at the picture result, duplicate recognition problems occur. This problem can be solved by properly raising the threshold values in the configuration file. Threshold values should be determined by testing, but it is easy to determine between 0.6 and 0.9.
``` cmake
[class-attrs-all]
nms-iou-threshold=0.7
pre-cluster-threshold=0.7 
```

[Source Here](https://spyjetson.blogspot.com/2020/08/xavier-nx-deepstream-50-5-running.html)
[link](https://spyjetson.blogspot.com/2020/08/xavier-nx-deepstream-50-1.html)