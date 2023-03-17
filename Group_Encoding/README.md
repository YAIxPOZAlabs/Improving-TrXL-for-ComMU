# Group Encoding for ComMU: Dataset for Combinatorial Music Generation [from](https://github.com/POZAlabs/ComMU-code)

## Getting Started
- Note : This Project requires python version `3.8.12`. Set the virtual environment if needed.
### Setup
1. Clone this repository
2. Install required packages
    ```
    pip install -r requirements.txt
    ```
### Download the Data
```
cd dataset && ./download.sh && cd ..
```

## Training
```
python3 -m torch.distributed.launch --nproc_per_node=4 ./train_grouped.py --data_dir ./dataset/output_npy --work_dir ./workdir
```

## Generating
- generation involves choice of metadata, regarding which type of music(midi file) we intend to generate. the example of command is showed below.
    ```
python3 generate.py \
--checkpoint_dir ./checkpoints/checkpoint_best.pt \
--output_dir ./output_dir \
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

## Checkpoint File - this is our model's checkpoint!
[Download](https://drive.google.com/file/d/1pqyD35PSbslGCkQK5Tbqg1swcIksQJRD/view?usp=sharing)

## License
ComMU dataset is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). It is provided primarily for research purposes and is prohibited to be used for commercial purposes.
