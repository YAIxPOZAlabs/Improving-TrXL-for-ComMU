# ComMU with Transformer based Conditional VAE

[Notion](https://www.notion.so/binne/Transformer_CVAE-17bffaac1ca34e73b4042ec8d80921f8?pvs=4)

<p align='center'>
<img src="https://user-images.githubusercontent.com/68505714/225926339-0a896169-c513-4a60-827d-84874e36c594.png" alt="CVAE" width="400" height="600">
</p>

## Getting Started
- Note : This Project requires python version `3.8.12` and `pytorch 1.13.0`. Set the virtual environment if needed.
### Setup
1. Clone this repository
2. Install required packages
```
pip install -r requirements.txt
```

This is preprocessed data. You don't need to preprocess anymore.
```
cd dataset && ./download.sh && cd ..
```

## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 ./trainCVAE.py --data_dir ./dataset/output_npy --work_dir ./workdir
```

## Generating
- generation involves choice of metadata, regarding which type of music(midi file) we intend to generate. the example of command is showed below.
```
python3 generateCVAE.py \
--checkpoint_dir {./working_directory/checkpoint_best.pt} \
--output_dir {./output_dir} \
--bpm 70 \
--audio_key aminor \
--time_signature 4/4 \
--pitch_range mid_high \
--num_measures 8 \
--inst acoustic_piano \
--genre newage \
--min_velocity 60 \
--max_velocity 80 \
--track_role main_melody \
--rhythm standard \
--chord_progression Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G-F-F-F-F-F-F-F-F-E-E-E-E-E-E-E-E-Am-Am-Am-Am-Am-Am-Am-Am-G-G-G-G-G-G-G-G-F-F-F-F-F-F-F-F-E-E-E-E-E-E-E-E \
--num_generate 3
```

## Checkpoints
- Sadly, No checkpoint is available in this project..
