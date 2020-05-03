# Emotic 

Humans use their facial features or expressions to convey how they feel, such as a person may smile when happy and scowl when angry. Historically, computer vision research has focussed on analyzing and learning these facial features to recognize emotions. 
However, these facial features are not universal and vary extensively across cultures and situations. 


<img src="https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/face.jpg">    <img src="https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/full_scene.jpg" width="400">
###### Fig 1: a) (Facial feature) The person looks angry or in pain b) (Whole scene) The person looks elated. 

A scene context, as shown in the figure above, can provide additional information about the situations. This project explores the use of context in recognizing emotions in images. 

## Pipeline 

The project uses the EMOTIC dataset and follows the methodology as introduced in the paper [*'Context based emotion recognition using EMOTIC dataset'*](https://arxiv.org/pdf/2003.13401.pdf)*.

![Pipeline](https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/pipeline%20model.jpg "Model Pipeline") 
###### Fig 2: Model Pipeline ([Image source](https://arxiv.org/pdf/2003.13401.pdf))

Two feature extraction modules first extract features over an image. These features are then used by a third module to predict the continuous dimensions (valence, arousal and dominance) and the discrete emotion categories.

## Usage 

Download the [Emotic dataset](https://drive.google.com/open?id=0B7sjGeF4f3FYQUVlZ3ZOai1ieEU) and [annotations](https://1drv.ms/u/s!AkYHbdGNmIVCgbYJxp1EtUplH6BhSw?e=VUP26u) and prepare the directory following the below structure: 
```
├── ...
│   ├── emotic
│   |    ├── ade20k
│   |    ├── emodb_small
│   |    ├── framesdb
│   |    ├── mscoco 
│   ├── Annotations
│   |    ├── Annotations.mat
```

1. To convert annotations from mat object to csv files and preprocess the data: 

```
> python mat2py.py --data_dir proj/data/emotic19 --generate_npy
```
* data_dir: Path of the directory containing the emotic and annotations folder as described in the above data directory structure. 
* generate_npy: Argument to specify to generate npy files (later used for training and testing) along with CSV files. If not passed only CSV files are generated. 

2. To train the model: 

```
> python main.py --mode train --data_path proj/data/emotic_pre --experiment_path proj/debug_exp
```
* mode: Mode to run the main file.
* data_path: Path of the directory which contains the preprocessed data and CSV files generated in the first step.  
* experiment_path: Path of the experiment directory. The directory will save the results, models and logs. 

3. To test the model: 

```
> python main.py --mode test --data_path proj/data/emotic_pre --experiment_path proj/debug_exp
```
* mode: Mode to run the main file.
* data_path: Path of the directory which contains the preprocessed data and CSV files generated in the first step.  
* experiment_path: Path of the experiment directory. The directory will save the results, models and logs. 



## Acknowledgements

* [Places365-CNN](https://github.com/CSAILVision/places365) 
* [Pytorch-Yolo](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### Context Based Emotion Recognition using Emotic Dataset 
_Ronak Kosti, Jose Alvarez, Adria Recasens, Agata Lapedriza_ <br>
[[Paper]](https://arxiv.org/pdf/2003.13401.pdf) [[Project Webpage]](http://sunai.uoc.edu/emotic/) [[Authors' Implementation]](https://github.com/rkosti/emotic)

```
@article{kosti2020context,
  title={Context based emotion recognition using emotic dataset},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  journal={arXiv preprint arXiv:2003.13401},
  year={2020}
}
```

## Author 
[Abhishek Tandon](https://github.com/Tandon-A)


