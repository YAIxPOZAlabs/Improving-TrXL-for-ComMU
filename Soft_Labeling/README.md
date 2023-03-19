# Soft Labeling

## Getting Started
- Note : This Project requires python version `3.8.12`. Set the virtual environment if needed.

### Setup
1. Clone this repository and change directory
```
git clone https://github.com/YAIxPOZAlabs/Generation.git
cd Generation/Soft_Labeling
```

2. Install required packages
```
pip install -r requirements.txt
```

## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 ./train.py --data_dir ./dataset/output_npy --work_dir ./workdir --soft_argument [0, 0.15, 0.7, 0.15, 0]
```

## Checkpoint File
This is our model(soft labeling)'s checkpoint!
[Download](https://drive.google.com/file/d/1DPA3Gsc_mT4myhZzdXsdMUNptHiYGAsE/view?usp=share_link)

## License
ComMU dataset is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). It is provided primarily for research purposes and is prohibited to be used for commercial purposes.
