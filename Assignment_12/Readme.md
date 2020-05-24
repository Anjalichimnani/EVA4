# Assignment 12
## 1. Tiny ImageNet Implementation using ResNet18 Network, Transformation, Step LR Scheduler 

Tiny Imagenet Data with 200 classes is learnt via ResNet18 Network. The code is modular and all the components are parameterized for flexibility and maintenance. Transformations and Regularization are performed on Train set and Transformations on Test set are simple. SGD Optimizer with Momentum and StepLR Scheduler is used during training to achieve at required validation accuracy

### Model Analysis: 
- Target: 
  - Architecture - [ResNet18!](https://arxiv.org/abs/1512.03385)
  - Validation Accuracy >= 50%
  - Epochs - 50

- Results:
  - Best Train Accuracy: > 99.98%
  - Best Test Accuracy:  > 53.82%
  - Final Test Accuracy: = 52.31%
 
Analysis:
  - Epochs used: 50
  - Total Parameters used: 11,271,432
  - RandomRotation along with Normalize and ToTensor are applied.
  
  
### Tiny ImageNet code is at [Tiny ImageNet ResNet Implementation!](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_12/EVA_S12_TinyImageNet.ipynb)



## 2. Custom Dataset:
Creation of a custom dataset with 50 Dog images and consequently annotated using [Image Annotation Tool!] (http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). The Annotation tool generates output (in JSON/COCO/CSV) with multiple Attibutes that could be parsed to retrieve desired information. 

### Attribute Description of JSON output form Annotation Tool:
The output is a single JSON Object with all images as keys in the object. 
The Image keys have name in format as  <ImageName><ImageSize>.<ImageFormat>

```"images_01.jpg8323":```

Each Image instance/Element is an object in itself with different attributes: 
```"images_01.jpg8323": {
		"filename": "images_01.jpg",
		"size": 8323,
		"regions": [],
        "file_attributes":{}
}```

The first Attribute is the FileName: The Image file name being annotated
```"filename": "images_01.jpg"```

The Next Attribute is the size of the image (not segregated into width and height). 
```"size": 8323```

Region forms the next attribute which is an array of objects, which contains the information for all annotations. Each annotated element forms one array element with 2 Attributes: Shape Attributes and Region Attibutes. 
Shape Attribute provides information as shape name(square/rectangle/pentagon), center of the annotation (square/rectangle/any shape) and Shape dimensions (Width/Height in case of square/rectangle). 
```"shape_attributes": {
					"name": "rect",
					"x": 35,
					"y": 4,
					"width": 152,
					"height": 163
}```

Region Attributes provides other descriptive information as required for the annotated region as Annotated Object Name, Type of object (class), Image quality (any description required). 
```"region_attributes": {
					"name": "DOG_1",
					"type": "dog",
					"image_quality": {
						"good": true,
						"frontal": true,
						"good_illumination": true
}```
Thus, ```regions``` attribute is an array of the object with shape and region attributes for all annotated objects in the image as:
```"regions": [ {...}, {...}, ..]```

The Last Attribute is the File Attributes which provides details for the image as any caption, url, public domain and can be used to descrive characteristics of whole image.
```"file_attributes": {
			"caption": "Image with 2 dogs Overlap",
			"public_domain": "no",
			"image_url": ""
}```


### Custom Dog Dataset: 50 Images is present at [Dog Dataset!](https://github.com/Anjalichimnani/EVA4/tree/master/Assignment_12/data/dogs)

## The Annotations for the Dog dataset are present at [Annotated JSON!](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_12/data/dogs/Final_Dog_Annotations.json)

### KMeans Clustering for Anchor boxes for custom Dog Dataset [KMeans Cluster Implementation!](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_12/EVA_12_KMeans_Clustering.ipynb)
Code for implmenting the KMeans Cluster: 
```
def kmeans_clusters_wcss (X, seed_range, init, max_iter, n_init, random_state):
        wcss = []
        
        for i in range(1, seed_range + 1):
            kmeans = KMeans(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
		
        return wcss
        
Where, X is an array of Width and Height, seed_range is for the K values, max_iter is number of iterations for the KMeans cluster algorithm, and any random seed as state to the execution. Return is wcss which is the Within Cluster sum of squanre of distances of ecah point from the centroid calculated which could be used to derive the number of clusters for optimum anchor boxes. 
```

### References:
[Tiny ImageNet Dataset Download!](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
[ResNet!](https://github.com/kuangliu/pytorch-cifar)
[Image Annotation Tool!](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)
[KMeans Clustering!](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203)