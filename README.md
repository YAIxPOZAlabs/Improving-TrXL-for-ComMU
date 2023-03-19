<!-- HEADER START -->
<!-- src: https://github.com/kyechan99/capsule-render -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:B993D6,100:8CA6DB&height=220&section=header&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=40&text=Improving%20Transformer-XL" alt="header" />
</a></p>
<h3 align="center">Improving Transformer-XL for Music Generation</h3>

<p align="center"><a href="https://github.com/YAIxPOZAlabs"><img src="https://raw.githubusercontent.com/YAIxPOZAlabs/MuseDiffusion/master/assets/figure00_logo.png" width=50% height=50% alt="logo"></a></p>

<p align="center">This project was carried out by <b><a href="https://github.com/yonsei-YAI">YAI 11th</a></b>, in cooperation with <b><a href="https://github.com/POZAlabs">POZAlabs</a></b>.</p>

<p align="center">
<br>
<a href="mailto:dhakim@yonsei.ac.kr">
    <img src="https://img.shields.io/badge/-Gmail-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Gmail"/>
</a>
<a href="https://www.notion.so/dhakim/1e7dc19fd1064e698a389f75404883c7?pvs=4">
    <img src="https://img.shields.io/badge/-Project%20Page-000000?style=flat-square&logo=notion&logoColor=white" alt="NOTION"/>
</a>
<a href="#">
    <img src="https://img.shields.io/badge/-Full%20Report-dddddd?style=flat-square&logo=latex&logoColor=black" alt="REPORT"/>
</a>
</p>
<br>
<hr>
<!-- HEADER END -->

# Improving Transformer-XL for Music Generation
YAI x POZAlabs 산학협력 1팀 <br>
NLP model for music generation <br>

# Members
👑조정빈<br>
박민수<br>
박수빈<br>
김성준<br>
김산<br>
김민서<br>
</br>

## Requirements
### Versions (Recommended)

<br>

# Building on Transformer-XL

## 0. Baseline (Transformer-XL) - [Link](https://github.com/kimiyoung/transformer-xl)
 Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

## 1. Group Encoding - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/Group_Encoding)
For a vanila transformer-XL model, it inputs tokens in a 1d sequence and adds Positional Encoding to give the model information about the position between tokens. In this setting, the model learns about the semantics of the data as well as the structure of the MIDI data. However, as there is an explicit pattern when encoding MIDI data in to sequence of tokens, we propose a Group Encoding method that injects an inductive bias about the explicit structure of the token sequence to the model. This not only keeps the model from inferencing strange tokens in strange positions, it also allows the model to generate 4 tokens in a single feed forward, boosting the training speed as well as the inference speed of the model.


<p align="center">
<!--<img width="500" src="https://user-images.githubusercontent.com/68505714/225891370-6de30c11-446a-44c4-885f-8896b9a0aa2b.png" alt="group">-->
</p>


<p align="center">
<img width="500" alt="GE" src="https://user-images.githubusercontent.com/73946308/225978762-90ec5154-4344-4e04-9a0d-caeb2042edd5.png">
</p>

<p align="center">
<img width="1089" alt="GE_1" src="https://user-images.githubusercontent.com/73946308/226155426-98fe608d-be28-4160-ad48-cf40c65d0728.png">
</p>


sampled audio

https://user-images.githubusercontent.com/68505714/225961723-93262632-abc2-41d0-8b8f-de2bafd5cc57.mov



## 2. Soft Labeling - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/Soft_Labeling)
To prevent overfitting of the model, techniques such as soft labeling are often used. We apply soft labeling on velocity, duration, and position information, so it can be flexibly predicted. For example, if the target of the token value is 300, the logit is reconstructed by referring to the logit value of the 298/299/301/302 token. As a result of using soft labeling, we confirm that the token appears more flexible than baseline.

<p align="center">
<img width="517" alt="softlabeling" src="https://user-images.githubusercontent.com/73946308/226155428-60fceb1c-6e78-436a-b8fc-edf231c8c695.png">
</p>

## 3. Gated Transformer-XL - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/Gated_Transformer-XL)

<br>

# Dataset

### ComMU (POZAlabs)
https://github.com/POZAlabs/ComMU-code


<br>

# Metrics
To evaluate generation models we have to generate data with trained models and depending on what metrics we want to use, the generation proccess differ. 
Please refer to the explanations below to generate certain samples needed for evaluation.


## Classification Accuracy Score - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/CAS)
 Evaluating Generative Models is an open problem and for Music generation has not been well defined. Inspired by 'Classification Accuracy Score for Conditional Generative Models' we use CAS as an evaluation metric for out music generation models. THe procedure of our CAS is the following
 
 0. Train a Music Generation Model with ComMU train set which we call 'Real Dataset'
 1. Generate samples and form a dataset which we call 'Fake Dataset'
 2. Train a classification model with 'Real Dataset' which we call 'Real Model'
 3. Train a classification model with 'Fake Dataset' which we call 'Fake Model'
 4. For each lable (meta data) we compare the performance of 'Fake Model' and 'Real Model' on ComMU validation set
 
 
 
<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/73946308/225903685-b8680fe4-4c31-4456-88d0-501cbb1d509a.png"/>
</p>

- To compute Classicication Accuracy Score of Generated Music conditioned with certain meta data 

to generate samples, run
```
$python generate_CAS.py --checkpoint_dir {./model_checkpoint} --meta_data {./meta_data.csv}
```
to compute CAS for certain meta data as label, run
```
$ python compute_CAS.py --midi_dir {./data.npy} --meta_dir {./meta.npy} --meta_num {meta_num}
```
to compute CAS for all meta data as label, run
```
$ python compute_CAS_all.py --midi_dir {./data.npy} --meta_dir {./meta.npy}
```


## Diversity
- To compute the Diversity of Generated Music conditioned with certain meta data

to generate sampels, run
```
$ python generate_diversity.py --checkpoint {./model_checkpoint}
```
to compute Diversity, run
```
$ python comput_diversity.py 
```

### Controllability


## Skills
Frameworks <br><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> 
