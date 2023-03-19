# wav2vec2-asr-ultimate-german

## Training
python train.py --output_dir \out --save_total_limit=2 --evaluation_strategy="steps"

## Sources
https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/run_common_voice.py
https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py