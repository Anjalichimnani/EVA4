# Assignment 13
## 1. Yolo Object Detection using OpenCV in Python

Performing object detection using YoloV3 Open CV implementation in Python. Using a custom image and classes in coco data set, tagging a custom image with the objects detected. The code reference is [YoloV3 OpenCV!](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

Custom Annotated Image using YoloV3 Open CV Implementation:
![Image] (https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_13/data/coco/custom_image.jpg)
  
### YoloV3 implementation with OpenCV is at [YoloV3 OpenCV!](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_13/YoloV3_OpenCV_Custom.ipynb)


## 2. YoloV3 Object Localization: Video Annotation:
The YoloV3 implementation is learnt on a Custom Class -> Kung Fu Panda. 
The Learning is performed using transfer learning, first learning on Small Coco data set and then, learning on 500 (~550) images of Kung Fu Panda class. 
The Custom Trained model is run on varied (3) Videos with different characteristics to obtain the annotations for the custom class.  

### Custom Kung Fu Panda Dataset: 550 Images is present at [Kung Fu Panda Dataset!](https://github.com/Anjalichimnani/EVA4/tree/master/Assignment_13/data/kungfupanda/Annotation/Images/kungfupanda)

### The Annotations for the Dog dataset are present at [Annotated txt!](https://github.com/Anjalichimnani/EVA4/tree/master/Assignment_13/data/kungfupanda/Annotation/Labels/kungfupanda)

### YoloV3 implementation for Custom Class is present at [YoloV3 Custom Class!](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_13/YoloV3_Localization.ipynb)

### Outcome of the YoloV3 Detection on videos: 
[Kung Fu Panda Final Fighting Scene!](https://youtu.be/yk3cm6nScDw)

[Kung Fu Panda Beginning Scene!](https://youtu.be/2RDKrh3Eo3s)

[Kung Fu Panda Openning Battle Scene!](https://youtu.be/zcmLTy7knLw)

### Image Outcome of the YoloV3 Detection on videos: 
[Kung Fu Panda Final Fighting Scene!](/data/kungfupanda/out_out)

[Kung Fu Panda Beginning Scene!](/data/kungfupanda/out_out_beginning)

[Kung Fu Panda Openning Battle Scene!](/data/kungfupanda/out_out_openning_battle)

### Sample Annotated Images Outcomes from the YoloV3 Implementation
Sample Annotated Image 01:
![Image](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_13/data/kungfupanda/Annotated_Image_Out_01.jpg)

Sample Annotated Image 02:
![Image](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_13/data/kungfupanda/Annotated_Image_Out_02.jpg)

Sample Annotated Image 03:
![Image](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_13/data/kungfupanda/Annotated_Image_Out_03.jpg)


## References:
[YoloV3 Open CV Implementation](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

[YoloV3](https://github.com/theschoolofai/YoloV3)

[YoloV3 Annotation Tool](https://github.com/miki998/YoloV3_Annotation_Tool)

[YoloV3 Custom Class](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)

[Extract Frames](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)