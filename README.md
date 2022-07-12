# Emotic 

Humans use their facial features or expressions to convey how they feel, such as a person may smile when happy and scowl when angry. Historically, computer vision research has focussed on analyzing and learning these facial features to recognize emotions. 
However, these facial features are not universal and vary extensively across cultures and situations. 


<img src="https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/face.jpg">    <img src="https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/full_scene.jpg" width="400">
###### Fig 1: a) (Facial feature) The person looks angry or in pain b) (Whole scene) The person looks elated. 

A scene context, as shown in the figure above, can provide additional information about the situations. This project explores the use of context in recognizing emotions in images. 

## Pipeline 

The project uses the EMOTIC dataset and follows the methodology as introduced in the paper *['Context based emotion recognition using EMOTIC dataset'](https://arxiv.org/pdf/2003.13401.pdf)*.

![Pipeline](https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/pipeline%20model.jpg "Model Pipeline") 
###### Fig 2: Model Pipeline ([Image source](https://arxiv.org/pdf/2003.13401.pdf))

Two feature extraction modules first extract features over an image. These features are then used by a third module to predict the continuous dimensions (valence, arousal and dominance) and the discrete emotion categories.

## Emotic Dataset 

The Emotic dataset can be used only for **non-commercial research and education purposes**.
Please, fill out the following form to request access to the dataset and the corresponding annotations.

[Access Request for EMOTIC](https://forms.gle/wvhComeDHwQPD6TE6)

## Usage
Download the Emotic dataset & annotations, and prepare the directory following the below structure: 
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
* experiment_path: Path of the experiment directory. Models stored in the the directory are used for testing. 

4. To perform inference: 

```
> python main.py --mode inference --inference_file proj/debug_exp/inference_file.txt --experiment_path proj/debug_exp
```
* mode: Mode to run the main file.
* inference_file: Text file specifying images to perform inference. A row is: 'full_path_of_image x1 y1 x2 y2', where (x1,y1) and (x2,y2) specify the bounding box. Refer [sample_inference_list.txt](https://github.com/Tandon-A/emotic/blob/master/sample_inference_list.txt).
* experiment_path: Path of the experiment directory. Models stored in the the directory are used for inference.     
  
  
You can also train and test models on Emotic dataset by using the [Colab_train_emotic notebook](https://github.com/Tandon-A/emotic/blob/master/Colab_train_emotic.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tandon-A/emotic/blob/master/Colab_train_emotic.ipynb)

The **trained models and thresholds** to use for inference purposes are availble [here](https://drive.google.com/drive/folders/1e-JLA7V73CQD5pjTFCSWnKCmB0gCpV1D?usp=sharing). 

## Results 

![Result GIF 1](https://github.com/Tandon-A/emotic/blob/master/assets/eld11_gif2.gif "Result GIF 1")

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


