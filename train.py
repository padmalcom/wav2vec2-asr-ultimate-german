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
from model import Wav2Vec2ForCTCnCLS
import re
import librosa
import os
import torch
from ctctrainer import CTCTrainer
from orthography import Orthography
from datacollator import DataCollatorCTCWithPadding
from tokenizer import build_tokenizer

@dataclass
class DataTrainingArguments:
	dataset_config_name = None
	train_split_name = "train"
	validation_split_name = "validation"
	target_text_column = "sentence"
	speech_file_column = "file"
	age_column = "age"
	target_feature_extractor_sampling_rate = False
	max_duration_in_seconds = None
	orthography = "librispeech"
	overwrite_cache = False
	preprocessing_num_workers = 1
	output_dir = "output/tmp"
	
@dataclass
class ModelArguments:
	#model_name_or_path = "facebook/wav2vec2-base-960h"
	model_name_or_path = "facebook/wav2vec2-large"
	cache_dir = "cache/"
	freeze_feature_extractor = False
	verbose_logging = False
	alpha = 0.1
	tokenizer = "facebook/wav2vec2-base-960h"
	
@dataclass
class TrainingArgs:
	output_dir = "output/tmp"
	per_device_train_batch_size = 2
	full_determinism = False
	seed = 42
	skip_memory_metrics = True
	# TODO: Add other arguments
				
if __name__ == "__main__":
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	
	os.makedirs(training_args.output_dir, exist_ok=True)
	
	# TODO: Load checkpoint
	
	#base_path = os.path.join("E:", os.sep, "Datasets", "common-voice-12")
	base_path = os.path.join("common-voice-12")
	
	orthography = Orthography()
	orthography.tokenizer = model_args.tokenizer
	print("Ortho: ", orthography.tokenizer)
	
	# Load dataset
	dataset = datasets.load_dataset('csv', data_files={'train': os.path.join(base_path, 'train.csv'), 'test': os.path.join(base_path, 'test.csv')})
	print("Dataset:", dataset)
	print("Test:", dataset['test'])
	print("Test0:", dataset['test'][0])
	
	# create processor
	feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
		model_args.model_name_or_path, cache_dir=model_args.cache_dir
	)
	#tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
	#	orthography.tokenizer,
	#	cache_dir=model_args.cache_dir,
	#	do_lower_case=orthography.do_lower_case,
	#	word_delimiter_token=orthography.word_delimiter_token
	#)
	tokenizer = build_tokenizer(training_args.output_dir, dataset['train'], data_args.preprocessing_num_workers)
	processor = Wav2Vec2Processor(feature_extractor, tokenizer)
	

	
	# create label maps
	cls_emotion_label_map = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
	cls_age_label_map = {'teens':0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5, 'seventies': 6, 'eighties': 7}
	cls_gender_label_map = {'female': 0, 'male': 1}
	
	print("vocab size: ", len(processor.tokenizer))
	
	# Load model
	model = Wav2Vec2ForCTCnCLS.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		gradient_checkpointing=True,
		vocab_size=len(processor.tokenizer),
		cls_len=len(cls_age_label_map),
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
			print("Example:", example)
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
			print("Batch age:", batch[data_args.age_column])
			#cls_labels = list(map(lambda e: cls_label_map[e], batch["emotion"]))
			cls_labels = list(map(lambda e: cls_age_label_map[e], batch[data_args.age_column]))# batch[data_args.age_column]
			with processor.as_target_processor():
				batch["labels"] = processor(batch[data_args.target_text_column]).input_ids
			for i in range(len(cls_labels)):
				batch["labels"][i].append(cls_labels[i]) # batch["labels"] element has to be a single list
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
		cls_pred_logits = pred.predictions[1]
		cls_pred_ids = np.argmax(cls_pred_logits, axis=-1)
		total = len(pred.label_ids[1])
		correct = (cls_pred_ids == pred.label_ids[1]).sum().item() # label = (ctc_label, cls_label)

		ctc_pred_logits = pred.predictions[0]
		ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
		pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
		ctc_pred_str = processor.batch_decode(ctc_pred_ids)
		# we do not want to group tokens when computing the metrics
		ctc_label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)
		if logger.isEnabledFor(logging.DEBUG):
			for reference, predicted in zip(label_str, pred_str):
				logger.debug(f'reference: "{reference}"')
				logger.debug(f'predicted: "{predicted}"')
				#if orthography.untransliterator is not None:
				#	logger.debug(f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
				#	logger.debug(f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

		wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)
		return {"acc": correct/total, "wer": wer, "correct": correct, "total": total, "strlen": len(ctc_label_str)}
		
	if model_args.freeze_feature_extractor:
		model.freeze_feature_extractor()
		
	print("Val dataset:", val_dataset)
	trainer = CTCTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		tokenizer=processor.feature_extractor,
	)
	
	trainer.train()
	trainer.save_model() 