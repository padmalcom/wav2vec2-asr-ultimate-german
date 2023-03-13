from transformers import (
	Wav2Vec2FeatureExtractor,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2Processor
)
import librosa
from datasets import Dataset
from datasets import disable_caching
import numpy as np
import torch.nn.functional as F
import torch
from model import Wav2Vec2ForCTCnCLS
from ctctrainer import CTCTrainer
from orthography import Orthography
from datacollator import DataCollatorCTCWithPadding

disable_caching()

cls_age_label_map = {'teens':0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5, 'seventies': 6}
model_path = "out/"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)#, cache_dir=model_args.cache_dir)

orthography = Orthography()
orthography.tokenizer = "facebook/wav2vec2-base-960h"
	
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
	orthography.tokenizer,
	#cache_dir=model_args.cache_dir,
	do_lower_case=orthography.do_lower_case,
	word_delimiter_token=orthography.word_delimiter_token
)
processor = Wav2Vec2Processor(feature_extractor, tokenizer)

model = Wav2Vec2ForCTCnCLS.from_pretrained(
	model_path,
	#cache_dir=model_args.cache_dir,
	gradient_checkpointing=False,
	vocab_size=len(processor.tokenizer),
	cls_len=len(cls_age_label_map),
	alpha=0.1,
)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, audio_only=True)

def genDataset():
	yield {"file": "audio.wav"}

target_sr = 16000

def prepare_dataset_step1(example):
	example["speech"], example["sampling_rate"] = librosa.load(example["file"], sr=target_sr)
	return example
	
def prepare_dataset_step2(batch):
	batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
	return batch
	
val_dataset = Dataset.from_generator(genDataset)
val_dataset = val_dataset.map(prepare_dataset_step1, load_from_cache_file=False)
val_dataset = val_dataset.map(prepare_dataset_step2, batch_size=2, batched=True, num_proc=1, load_from_cache_file=False)

print("Val dataset:", val_dataset)
		
trainer = CTCTrainer(
	model=model,
	data_collator=data_collator,
	#args=training_args,
	#compute_metrics=compute_metrics,
	#train_dataset=train_dataset,
	eval_dataset=val_dataset,
	tokenizer=processor.feature_extractor,
)

print('******* Predict ********')
data_collator.audio_only=True
predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")
print("predictions:", predictions, "labels:", labels, "metrics:", metrics)
logits_ctc, logits_cls = predictions
pred_ids = np.argmax(logits_cls, axis=-1)
pred_probs = F.softmax(torch.from_numpy(logits_cls).float(), dim=-1)
print(val_dataset)
with open("prediction.txt", 'w') as f:
	for i in range(len(pred_ids)):
		f.write(val_dataset[i]['file'].split("/")[-1] + " " + str(len(val_dataset[i]['input_values'])/16000) + " ")
		pred = pred_ids[i]
		f.write(str(pred)+' ')
		for j in range(4):
			f.write(' ' + str(pred_probs[i][j].item()))
		f.write('\n')
f.close()