<!-- HEADER START -->
<!-- src: https://github.com/kyechan99/capsule-render -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:B993D6,100:8CA6DB&height=220&section=header&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=40&text=Improving%20Transformer-XL" alt="header" />
</a></p>
%20<h3 align="center">Improving Transformer-XL for Music Generation</h3>
<p align="center"><a href="https://github.com/YAIxPOZAlabs"><img src="assets/figure00_logo.png" width=50% height=50% alt="logo"></a></p>
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

## 0. Baseline (Transformer-XL)
 Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

## 1. Group Encoding

## 2. Soft Labeling


## 3. Gated Transformer-XL

<br>

# Dataset

### ComMU (POZAlabs)
https://github.com/POZAlabs/ComMU-code


<br>

## Metrics
To evaluate generation models we have to generate data with trained models and depending on what metrics we want to use, the generation proccess differ. 
Please refer to the explanations below to generate certain samples needed for evaluation.


### Classification Accuracy Score
<img width="80%" src="https://user-images.githubusercontent.com/73946308/225903685-b8680fe4-4c31-4456-88d0-501cbb1d509a.png"/>


- To compute Classicication Accuracy Score of Generated Music conditioned with certain meta data 

to generate samples, run
```
$python generate_CAS.py --checkpoint {./model_checkpoint} --meta_data {./meta_data.csv}
```
to compute CAS, run
```
$ python compute_CAS.py --data_dir {./data} --meta_name {KEY}
```

### Diversity
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
