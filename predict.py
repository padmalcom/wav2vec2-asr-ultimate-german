from transformers import (
	Wav2Vec2FeatureExtractor,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2Processor
)
import os
import librosa
from datasets import Dataset
from datasets import disable_caching
import numpy as np
import torch.nn.functional as F
import torch
from model import Wav2Vec2ForCTCnCLS
from ctctrainer import CTCTrainer
from datacollator import DataCollatorCTCWithPadding

disable_caching()

cls_age_label_map = {'teens':0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5, 'seventies': 6, 'eighties': 7}
cls_age_label_class_weights = [0] * len(cls_age_label_map)

cls_gender_label_map = {'female': 0, 'male': 1}
cls_gender_class_weights = [0] * len(cls_gender_label_map)
	
	
model_path = "ultimate-german/"

vocab_path = os.path.join(model_path, "tokenizer", "vocab.json")
tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor, tokenizer)

model = Wav2Vec2ForCTCnCLS.from_pretrained(
	model_path,
	vocab_size=len(processor.tokenizer),
	cls_len=len(cls_age_label_map),
	cls_weights=cls_age_label_class_weights,
	alpha=0.1,
)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, audio_only=True)
	
pred_data = {'file': ['audio2.wav']}

target_sr = 16000

def prepare_dataset_step1(example):
	example["speech"], example["sampling_rate"] = librosa.load(example["file"], sr=target_sr)
	return example
	
def prepare_dataset_step2(batch):
	batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
	return batch
	
val_dataset = Dataset.from_dict(pred_data)
val_dataset = val_dataset.map(prepare_dataset_step1, load_from_cache_file=False)
val_dataset = val_dataset.map(prepare_dataset_step2, batch_size=2, batched=True, num_proc=1, load_from_cache_file=False)

print("Val dataset:", val_dataset)
		
trainer = CTCTrainer(
	model=model,
	data_collator=data_collator,
	eval_dataset=val_dataset,
	tokenizer=processor.feature_extractor,
)

print('******* Predict ********')
data_collator.audio_only=True
predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")
print("predictions:", predictions, "labels:", labels, "metrics:", metrics)
logits_ctc, logits_cls = predictions
print("logits ctc:", logits_ctc, "logits cls:", logits_cls)

# process age classification
pred_ids_cls = np.argmax(logits_cls, axis=-1)
pred_age = pred_ids_cls[0]
age_class = [k for k, v in cls_age_label_map.items() if v == pred_age]
print("Predicted age: ", age_class[0])

# process token classification
pred_ids_ctc = np.argmax(logits_ctc, axis=-1)
pred_str = processor.batch_decode(pred_ids_ctc, output_word_offsets=True)
print("pred text: ", pred_str)