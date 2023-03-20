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

# Improving Transformer-XL for Music Generation 🎼
YAI x POZAlabs 산학협력 1팀 <br>
NLP model for music generation <br>

# Members 👋
<b> <a href="https://github.com/whwjdqls">조정빈</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp;whwjdqls99@yonsei.ac.kr 
<br>
<b>  <a href="">김민서</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp; min99830@yonsei.ac.kr  <br>
<b> <a href="https://github.com/mounKim">김산</a></b>&nbsp; :&nbsp; YAI 9th&nbsp; /&nbsp; nasmik419@yonsei.ac.kr <br>
<b> <a href="https://github.com/Tim3s">김성준</a></b>&nbsp; :&nbsp; YAI 9th&nbsp; /&nbsp;sjking0825@gmail.com <br>
<b> <a href="</br>https://github.com/0601p">박민수</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp;0601p@naver.com  <br>
<b><a href="https://github.com/starwh03">박수빈</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp;sallyna602@yonsei.ac.kr <br>
<br><br>
<hr>


# Getting Started 🔥
As there are different models and metrics, we recommand using seperate virtual envs for each. 
As each directory contains it's own "Getting Started", for clear instructions, please follow the links shown in each section.
```
Generation/
├─ CAS/
├─ Group_Encoding/
├─ Soft_Labeling/
├─ TransformerCVAE/

```
As for Baseline which is Transformer-XL trained on ComMU-Dataset, refer to the [ComMU-code](https://github.com/POZAlabs/ComMU-code) by [POZAlabs](https://github.com/POZAlabs)

# Building on Transformer-XL

## 0. Baseline (Transformer-XL) - [Link](https://github.com/kimiyoung/transformer-xl)
 Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
### Evaluation
#### Classifcation Accuracy Score
|             Lable(Meta)          |  Real Model | Fake Model | error rate | 
|-----------------------|----|----------|-----|
| BPM | 0.6291	| 0.6159 |0.0210 |
| KEY | 0.8781|  0.8781 |  0 | 
| TIMESIGNATURE | 0.9082 | 0.8925 | |
| PITCHRANGE | 0.7483 | 0.7090 | 0.0525 |
| NUMEIEROFMEASURE | 1.0 | 1.0 | |
| INSTRUMENT | 0.5858 | 0.5923| |
| GENRE | 0.8532 | 0.8427 | 0.0123 |
| MINVELOCITY | 0.4718 | 0.4482 | |
| MAXVELOCITY | 0.4718 | 0.4495 |  |
| TRACKROLE | 0.6500 | 0.5753 | 0.1149 |
| RHYTHM | 0.9934 | 0.9934 |   |

Normalized Mean CASD   : *0.0401*


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

### Evaluation 
#### Controllability and Diversity

|                       | CP | CV(Midi) | CV(Note) | CH | Diversity |
|-----------------------|----|----------|----------|----|-----------|
| Transformer XL w/o GE | 0.8585	| 0.8060 | 0.9847 | 0.9891 | 0.4100     |
| Transformer XL w GE   |   0.8493	| 0.7391 | 0.9821	| 0.9839	| 0.4113    |

#### Classification Accuracy Score
|             Lable(Meta)          |  Real Model | Fake Model   |error rate |
|-----------------------|----|----------|---|
| BPM | 0.6291	| 0.5910 |0.0606|
| KEY | 0.8781|  0.8532 |0.0284|
| TIMESIGNATURE | 0.9082 | 0.8951 | |
| PITCHRANGE | 0.7483 | 0.7195 | 0.0385|
| NUMEIEROFMEASURE | 1.0 | 1.0 |
| INSTRUMENT | 0.5858 | 0.5884|
| GENRE | 0.8532 | 0.8532 | 0 |
| MINVELOCITY | 0.4718 | 0.4364 |
| MAXVELOCITY | 0.4718 | 0.4560 | 
| TRACKROLE | 0.6500 | 0.5360 | 0.1754 |
| RHYTHM | 0.9934 | 0.9934 |  

Normalized Mean CASD : *0.0605*

#### Inference Speed
|                       | Inference time for Valset | Inference speed per sample | relative speed up|
|-----------------------|----|----------|---|
| Transformer XL w/o GE | 1189.4s | 1.558s per sample | X1 | 
| Transformer XL w GE   |   692.2s	| 0.907s per sample | X1.718  |

  
### Sampled Audio
  5 note sequences with shared and different meta data were sampled by the following conditions and mixed together. 
  - Shared meta data acrross 5 samples
    - audio_key : aminor
    - chord_progressions  : [['Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']]
    - time_signature  : 4/4
    - genre : cinematic
    - bpm : 120
    - rhythm : standard

- Different meta data for each instrument
    - riff string_violin mid_high standard 
    - main_melody string_ensemble mid_high 
    - sub_melody string_cello very_low 
    - pad acoustic_piano mid_low 
    - sub_melody brass_ensemble mid 




https://user-images.githubusercontent.com/68505714/225961723-93262632-abc2-41d0-8b8f-de2bafd5cc57.mov



## 2. Soft Labeling - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/Soft_Labeling)
To prevent overfitting of the model, techniques such as soft labeling are often used. We apply soft labeling on velocity, duration, and position information, so it can be flexibly predicted. For example, if the target of the token value is 300, the logit is reconstructed by referring to the logit value of the 298/299/301/302 token. As a result of using soft labeling, we confirm that the token appears more flexible than baseline.

<p align="center">
<img width="517" alt="softlabeling" src="https://user-images.githubusercontent.com/73946308/226155428-60fceb1c-6e78-436a-b8fc-edf231c8c695.png">
</p>

### Evaluation

#### Test set NLL

|              n-2         | n-1 | n | n+1 | n+2 | test NLL|
|-----------------------|----|----------|----------|----|-----------|
| 0 | 0 | 1 | 0 | 0  | 0.96     |
| 0.1 | 0.1 | 0.6 | 0.1 | 0.1  | 1.01     |
| 0 | 0.15 | 0.7 | 0.15 | 0  | 1.05     |
| 0.1 | 0.2 | 0.4 | 0.2 | 0.1  | 1.26     |



#### Classification Accuracy Score
|             Lable(Meta)          |  Real Model |  Fake Model  | error rate |
|-----------------------|----|----------|---|
| BPM | 0.6291	| 0.6133 |0.0251 |
| KEY | 0.8781|  0.8741 |0.0046 |
| TIMESIGNATURE | 0.9082 | 0.8990 | |
| PITCHRANGE | 0.7483 | 0.7195 | 0.0385|
| NUMEIEROFMEASURE | 1.0 | 1.0 | |
| INSTRUMENT | 0.5858 | 0.5740 | |
| GENRE | 0.8532 | 0.8440 |0.0108 |
| MINVELOCITY | 0.4718 | 0.4429 | |
| MAXVELOCITY | 0.4718 | 0.4429 |  |
| TRACKROLE | 0.6500 | 0.5661 | 0.1291 |
| RHYTHM | 0.9934 | 0.9934 | | 

Normalized Mean CASD: *0.0416*

## 3. Gated Transformer-XL - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/Gated_Transformer-XL)

<br>

# Dataset 🎶

### ComMU (POZAlabs)
 [ComMU-code](https://github.com/POZAlabs/ComMU-code) has clear instructions on how to download and postprocess ComMU-dataset, but we also provide a postprocessed dataset for simplicity.
To download preprocessed and postprocessed data, run

```
cd ./dataset && ./download.sh && cd ..
```

<br>

# Metrics 📋
To evaluate generation models we have to generate data with trained models and depending on what metrics we want to use, the generation proccess differ. 
Please refer to the explanations below to generate certain samples needed for evaluation.


## Classification Accuracy Score - [Link](https://github.com/YAIxPOZAlabs/Generation/tree/master/CAS)
 Evaluating Generative Models is an open problem and for Music generation has not been well defined. Inspired by 'Classification Accuracy Score for Conditional Generative Models' we use CAS as an evaluation metric for out music generation models. THe procedure of our CAS is the following
 
 0. Train a Music Generation Model with ComMU train set which we call 'Real Dataset'
 1. Generate samples and form a dataset which we call 'Fake Dataset'
 2. Train a classification model with 'Real Dataset' which we call 'Real Model'
 3. Train a classification model with 'Fake Dataset' which we call 'Fake Model'
 4. For each lable (meta data) we compare the performance of 'Fake Model' and 'Real Model' on ComMU validation set

 From the above procedure we can obtain CAS for a certain label (meta) we want to evaluate. If the difference between the accuracy of the 'Fake Model' and 'Real Model' is low, it means our generation model has captured the data distribution w.r.t the certain label well.
 For our experiments on vanila Transformer-XL, Transformer-XL with GE and Transformer-XL with SL, we calculate CAS on all 11 labels. However, some labels such as Number of Measure, Time Signature or Rhythm are usuited for evaluation. Therfore we select BPM, KEY, PITCH RANGE, GENRE and TRACK-ROLE and calculate the Normalized Mean Classification Accuracy Difference Score denoting it as CADS. We obtain CADS as the following. 
 
 <p align="center">
<img width="200" alt="GE" src="https://user-images.githubusercontent.com/73946308/226251962-bc3a2ad1-a2fa-4c1c-bd72-d758902c59cd.png">
</p>

where N is the number of labels(meta) that we think are relevent, in this case 5, and $R_i$ and $F_i$ denotes Real model accuracy for label num i and fake model accuracy for label num i respectively.

The following figure is the overal pipeline of CAS
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

## Citations
```bibtex
@misc{dai2019transformerxl,
      title={Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context}, 
      author={Zihang Dai and Zhilin Yang and Yiming Yang and Jaime Carbonell and Quoc V. Le and Ruslan Salakhutdinov},
      year={2019},
      eprint={1901.02860},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```bibtex
@misc{https://doi.org/10.48550/arxiv.1905.10887,
  doi = {10.48550/ARXIV.1905.10887},
  url = {https://arxiv.org/abs/1905.10887},
  author = {Ravuri, Suman and Vinyals, Oriol},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Classification Accuracy Score for Conditional Generative Models},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@inproceedings{hyun2022commu,
  title={Com{MU}: Dataset for Combinatorial Music Generation},
  author={Lee Hyun and Taehyun Kim and Hyolim Kang and Minjoo Ki and Hyeonchan Hwang and Kwanho Park and Sharang Han and Seon Joo Kim},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
}
```

