from commu.midi_generator.generate_pipeline import MidiGenerationPipeline
from commu.preprocessor.encoder import encoder_utils
from commu.preprocessor.utils.constants import (
    BPM_INTERVAL,
    DEFAULT_POSITION_RESOLUTION,
    DEFAULT_TICKS_PER_BEAT,
    VELOCITY_INTERVAL,
    SIG_TIME_MAP,
    KEY_NUM_MAP
)
from commu.preprocessor.encoder.event_tokens import base_event, TOKEN_OFFSET
import numpy as np
import pandas as pd
import miditoolkit
import argparse
import math
import re

def main(checkpoint_path, meta_data_path, eval_diversity=False, out_dir = './out_val'):
    pipeline = MidiGenerationPipeline()

    # Model Initialize
    pipeline.initialize_model(model_arguments={'checkpoint_dir': checkpoint_path})
    pipeline.initialize_generation()

    inference_cfg = pipeline.model_initialize_task.inference_cfg
    model = pipeline.model_initialize_task.execute()

    # Validation Meta Data load
    metas = pd.read_csv(meta_data_path)
    del metas['Unnamed: 0']

    val_npy_list = []
    num_notes = 0
    for idx, meta in enumerate(metas.iloc):
        input_ = dict(meta)
        if input_['num_measures'] == 5:
            input_['num_measures'] = 4
        if input_['num_measures'] == 9:
            input_['num_measures'] = 8
        if input_['num_measures'] == 17:
            input_['num_measures'] = 16

        input_['rhythm'] = 'standard'
        input_['output_dir'] = out_dir + '/' + str(idx)
        input_['num_generate'] = 10 if eval_diversity else 1
        input_['top_k'] = 32
        input_['temperature'] = 0.95
        input_['chord_progression'] = str('-'.join(eval(input_['chord_progressions'])[0]))
        input_.pop('chord_progressions')
        
        pipeline.preprocess_task.input_data = None
        encoded_meta = pipeline.preprocess_task.execute(input_)
        input_data = pipeline.preprocess_task.input_data

        pipeline.inference_task(
            model=model,
            input_data=input_data,
            inference_cfg=inference_cfg
        )

        sequences = pipeline.inference_task.execute(encoded_meta)
        pipeline.postprocess_task(input_data=input_data)
        pipeline.postprocess_task.execute(
            sequences=sequences,
        )
        val_npy_list.append(sequences)
        print(sequences)
        print(idx, '/', len(metas), flush=True)
    val_npy = np.array(val_npy_list)
    np.save(out_dir + '/val.npy', val_npy)
        
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--checkpoint_dir", type=str)
    arg_parser.add_argument("--val_meta_dir", type=str)
    arg_parser.add_argument("--eval_diversity", type=bool)
    arg_parser.add_argument("--out_dir", type=str)

    args = arg_parser.parse_args()

    main(args.checkpoint_dir, args.val_meta_dir, args.eval_diversity, args.out_dir)