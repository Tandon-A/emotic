# Emotic 

Humans use their facial features or expressions to convey how they feel, such as a person may smile when happy and scowl when angry. Historically, computer vision research has focussed on analyzing and learning these facial features to recognize emotions. 
However, these facial features are not universal and vary extensively across cultures and situations. 


<img src="https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/face.jpg">    <img src="https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/full_scene.jpg" width="400">
###### Fig 1: a) (Facial feature) The person looks angry or in pain b) (Whole scene) The person looks elated. 

A scene context, as shown in the figure above, can provide additional information about the situations. This project explores the use of context in recognizing emotions in images. 

## Pipeline 

The project uses the [EMOTIC dataset](https://drive.google.com/file/d/0B7sjGeF4f3FYQUVlZ3ZOai1ieEU/view) and follows the methodology as introduced in the paper *'Context based emotion recognition using EMOTIC dataset'*. ([paper](https://ieeexplore.ieee.org/document/8713881))

Two feature extraction modules first extract features over an image. These features are then used by a third module to predict the continuous dimensions (valence, arousal and dominance) and the discrete emotion categories. 

![Pipeline](https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/pipeline%20model.jpg "Emotic Pipeline")



## Acknowledgements

* [Emotions Recognition in Context](https://github.com/rkosti/emotic)
* [Places365-CNN](https://github.com/CSAILVision/places365) 

## Author 
[Abhishek Tandon](https://github.com/Tandon-A)


