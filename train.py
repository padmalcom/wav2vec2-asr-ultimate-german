from dataclasses import dataclass
from transformers import (
	TrainingArguments,
	HfArgumentParser,
	Wav2Vec2FeatureExtractor,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2Processor
)
import datasets
import evaluate
import pandas as pd
import re
import librosa
import os
import torch
import numpy as np
import json
from model import Wav2Vec2ForCTCnCLS
from ctctrainer import CTCTrainer
from datacollator import DataCollatorCTCWithPadding
#from tokenizer import build_tokenizer

@dataclass
class DataTrainingArguments:
	target_text_column = "sentence"
	speech_file_column = "file"
	age_column = "age"
	preprocessing_num_workers = 1
	output_dir = "output/tmp"
	
@dataclass
class ModelArguments:
	model_name_or_path = "facebook/wav2vec2-base"
	cache_dir = "cache/"
	freeze_feature_extractor = True
	alpha = 0.1
					
if __name__ == "__main__":
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	
	os.makedirs(training_args.output_dir, exist_ok=True)
	
	# TODO: Load checkpoint
	
	#base_path = os.path.join("E:", os.sep, "Datasets", "common-voice-12")
	base_path = os.path.join("common-voice-12")
		
	# Load dataset
	dataset = datasets.load_dataset('csv', data_files={'train': os.path.join(base_path, 'train.csv'), 'test': os.path.join(base_path, 'test.csv')})
	print("Dataset:", dataset)
	print("Test:", dataset['test'])
	print("Test0:", dataset['test'][0])
	
	#{'à': 0, 'ắ': 1, 'ś': 2, 'g': 3, 'ě': 4, 'ש': 5, 'ı': 6, 'y': 7, 'ć': 8, 'ô': 9, 'å': 10, 'נ': 11, ' ': 12, '̇': 13, 'ā': 14, 'č': 15, 'n': 16, 'ñ': 17, 'f': 18, 'a': 19, 'c': 20, 'o': 21, 'w': 22, 'd': 23, 'ø': 24, 'k': 25, 'x': 26, 'b': 27, '«': 28, 'ọ': 29, '‚': 30, 'ș': 31, 'q': 32, 'ë': 33, 'ț': 34, 'š': 35, 'ğ': 36, 'i': 37, 'h': 38, 'ב': 39, '»': 40, 'ł': 41, 'ʻ': 42, 'ę': 43, 'ş': 44, 'm': 45, 'ǐ': 46, 'ü': 47, 'ń': 48, 'â': 49, 'ß': 50, 'ä': 51, 'đ': 52, 'ö': 53, 'v': 54, 'p': 55, 't': 56, 'ע': 57, 's': 58, 'ă': 59, 'j': 60, 'z': 61, 'á': 62, 'e': 63, 'ý': 64, 'í': 65, 'ź': 66, 'ã': 67, 'ō': 68, 'ʿ': 69, 'l': 70, 'ú': 71, 'ứ': 72, 'ą': 73, 'u': 74, 'ó': 75, 'ê': 76, 'r': 77, 'é': 78, 'א': 79}
	
	#chars_to_ignore_regex = '[\,\¿\?\.\¡\!\-\;\:\"\“\%\‘\”\￼\…\’\ː\'\‹\›\`\´\®\—\→\–\„\à\ắ\ś\ě\5\ı\ć\ô\å\ā\č\ñ\ø\«\ọ\ș\ë\š\ğ\»\ב\ł\ę\ş\ǐ\ń\â\đ\ע\ă\á\ý\ź\ã\ō\ʿ\ú\ứ\ą\ó\ê\é\א\נ\ʻ\ț\ש\í\‚\š\đ\ś\ş\ł]'
	#chars_to_ignore_pattern = re.compile(chars_to_ignore_regex)
	
	german_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss'}
	
	def remove_special_characters(batch):
		batch["sentence"] = batch["sentence"].translate(german_char_map).lower()
		batch["sentence"] = batch["sentence"].encode('ascii', errors='ignore')
		#batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
		#batch["sentence"] = re.sub(r"[^a-zA-ZÄÜÖäüöß ]", '', batch["sentence"]).lower()
		return batch
		
	dataset = dataset.map(remove_special_characters)
	
	# create processor
	#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
	#tokenizer = build_tokenizer(training_args.output_dir, dataset['train'], data_args.preprocessing_num_workers)
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
	
	def extract_all_chars(batch):
		all_text = " ".join(batch["sentence"])
		vocab = list(set(all_text))
		return {"vocab": [vocab], "all_text": [all_text]}
		
	vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])
	vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
	vocab_dict = {v: k for k, v in enumerate(vocab_list)}
	print("vocab dict:", vocab_dict)
	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]
	vocab_dict["[UNK]"] = len(vocab_dict)
	vocab_dict["[PAD]"] = len(vocab_dict)
	print("vocal length:", len(vocab_dict))
	with open('vocab_new.json', 'w', encoding="utf8") as vocab_file:
		json.dump(vocab_dict, vocab_file)
	tokenizer = Wav2Vec2CTCTokenizer("./vocab_new.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", do_lower_case=True)		
	tokenizer.save_pretrained(os.path.join(training_args.output_dir, "tokenizer"))
	
	processor = Wav2Vec2Processor(feature_extractor, tokenizer)
		
	# create label maps and count of each label class
	cls_emotion_label_map = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
	cls_emotion_class_weights = [0] * len(cls_emotion_label_map)
	
	cls_age_label_map = {'teens':0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5, 'seventies': 6, 'eighties': 7}
	cls_age_label_class_weights = [0] * len(cls_age_label_map)
	
	cls_gender_label_map = {'female': 0, 'male': 1}
	cls_gender_class_weights = [0] * len(cls_gender_label_map)
	
	# count label sizes in train to balance classes
	df = pd.read_csv(os.path.join(base_path, 'train.csv'))
	
	df_age_count = df.groupby(['age']).count()
	for index, k in enumerate(cls_age_label_map):
		if k in df_age_count.index:
			cls_age_label_class_weights[index] = 1 - (df_age_count.loc[k]['file'] / df.size)
	print("Age label weights:", cls_age_label_class_weights)
	
	df_emotion_count = df.groupby(['emotion']).count()
	for index, k in enumerate(cls_emotion_label_map):
		if k in df_emotion_count.index:
			cls_emotion_class_weights[index] = 1 - (df_emotion_count.loc[k]['file'] / df.size)
	print("Emotion label weights:", cls_emotion_class_weights)

	df_gender_count = df.groupby(['gender']).count()
	for index, k in enumerate(cls_gender_label_map):
		if k in df_gender_count.index:
			cls_gender_class_weights[index] = 1 - (df_gender_count.loc[k]['file'] / df.size)
	print("Gender label weights:", cls_gender_class_weights)	
	
	print("vocab size: ", len(processor.tokenizer))
	
	# Load model
	model = Wav2Vec2ForCTCnCLS.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		gradient_checkpointing=True,
		vocab_size=len(processor.tokenizer),
		cls_len=len(cls_age_label_map),
		cls_weights=cls_age_label_class_weights,
		alpha=model_args.alpha,
	)
	
	# load metrics
	wer_metric = evaluate.load("wer")
	
	# preprocess data
	target_sr = 16000
	vocabulary_chars_str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
	vocabulary_text_cleaner = re.compile(
		f"[^\s{re.escape(vocabulary_chars_str)}]",
		flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
	)
	
	def prepare_example(example, audio_only=False):
		example["speech"], example["sampling_rate"] = librosa.load(os.path.join(base_path, "wavs", example[data_args.speech_file_column]), sr=target_sr)
		if audio_only is False:
			updated_text = " ".join(example[data_args.target_text_column].split()) # remove whitespaces
			updated_text = vocabulary_text_cleaner.sub("", updated_text)
			if updated_text != example[data_args.target_text_column]:
				example[data_args.target_text_column] = updated_text
		return example
		
	train_dataset = dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])['train']
	val_dataset = dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])['test']
	
	print(train_dataset)
	print(val_dataset)
	
	def prepare_dataset(batch, audio_only=False):
		batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
		if audio_only is False:
			cls_labels = list(map(lambda e: cls_age_label_map[e], batch[data_args.age_column]))
			with processor.as_target_processor():
				batch["labels"] = processor(batch[data_args.target_text_column]).input_ids
			for i in range(len(cls_labels)):
				batch["labels"][i].append(cls_labels[i]) # batch["labels"] element has to be a single list
		# the last item in the labels list is the cls_label
		return batch
		
	train_dataset = train_dataset.map(
		prepare_dataset,
		batch_size=training_args.per_device_train_batch_size,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
	)
	
	val_dataset = val_dataset.map(
		prepare_dataset,
		batch_size=training_args.per_device_train_batch_size,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
	)
	
	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
	
	def compute_metrics(pred):
		print("Metrics predictions: ", pred.predictions)
		cls_pred_logits = pred.predictions[1]
		cls_pred_ids = np.argmax(cls_pred_logits, axis=-1)
		print("cls pred ids:", cls_pred_ids)
		total = len(pred.label_ids[1])
		print("cls pred ids:", cls_pred_ids, "pred labels:", pred.label_ids[1])
		correct = (cls_pred_ids == pred.label_ids[1]).sum().item() # label = (ctc_label, cls_label)

		ctc_pred_logits = pred.predictions[0]
		ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
		pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
		ctc_pred_str = processor.batch_decode(ctc_pred_ids)
		# we do not want to group tokens when computing the metrics
		ctc_label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)
		print("ctc label:", ctc_label_str, "ctc prediction:", ctc_pred_str, "ctc pred ids:", ctc_pred_ids)


		wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)
		metric_res = {"acc": correct/total, "wer": wer, "correct": correct, "total": total, "strlen": len(ctc_label_str)}
		print("metric res:", metric_res)
		return metric_res
		
	if model_args.freeze_feature_extractor:
		model.freeze_feature_extractor()
		
	print("Val dataset:", val_dataset)
	print("Train dataset:", train_dataset)
	trainer = CTCTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		tokenizer=processor.feature_extractor
	)
	trainer.train()
	trainer.save_model(training_args.output_dir) 