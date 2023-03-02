from dataclasses import dataclass
from transformers import (
	TrainingArguments,
	HfArgumentParser,
	Wav2Vec2FeatureExtractor,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2Processor,
	Trainer
)
import datasets
import evaluate
from model import Wav2Vec2ForCTCnCLS
import re
import librosa
import os
import torch

@dataclass
class DataTrainingArguments:
	#dataset_name = "emotion" # TODO: change to custom
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
	model_name_or_path = "facebook/wav2vec2-base-960h"
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
	
@dataclass
class Orthography:
	do_lower_case = False
	vocab_file = None
	word_delimiter_token = "|"
	translation_table = {}
	words_to_remove = set()
	untransliterator = None
	tokenizer = None
	
class DataCollatorCTCWithPadding:
	def __init__(self, processor, padding):
		self.processor = processor
		self.padding = padding
		self.max_length = None
		self.max_length_labels = None
		self.pad_to_multiple_of = None
		self.pad_to_multiple_of_labels = None
		self.audio_only = False

	def __call__(self, features):
		# split inputs and labels since they have to be of different lenghts and need
		# different padding methods
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		if self.audio_only is False:
			label_features = [{"input_ids": feature["labels"][:-1]} for feature in features]
			cls_labels = [feature["labels"][-1] for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)
		if self.audio_only is False:
			with self.processor.as_target_processor():
				labels_batch = self.processor.pad(
					label_features,
					padding=self.padding,
					max_length=self.max_length_labels,
					pad_to_multiple_of=self.pad_to_multiple_of_labels,
					return_tensors="pt",
				)

			# replace padding with -100 to ignore loss correctly
			ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
			batch["labels"] = (ctc_labels, torch.tensor(cls_labels)) # labels = (ctc_labels, cls_labels)

		return batch
		
class CTCTrainer(Trainer):
	def _prepare_inputs(self, inputs):
		for k, v in inputs.items():
			#print("Key:", k, "value:", v)
			if isinstance(v, torch.Tensor):
				kwargs = dict(device=self.args.device)
				if self.deepspeed and inputs[k].dtype != torch.int64:
					kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
				inputs[k] = v.to(**kwargs)

			if k == 'labels': # labels are list of tensor, not tensor, special handle here
				#inputs[k] = inputs[k].to(**kwargs)
				new_labels = []
				for i in range(len(inputs[k])):
				#	kwargs = dict(device=self.args.device)
					#print("Input: ", inputs[k][i], "type: ", type(inputs[k][i]))
				#	if self.deepspeed and inputs[k][i].dtype != torch.int64:
				#		kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
				#	inputs[k][i] = inputs[k][i].to(**kwargs)
					new_labels.append(inputs[k][i].to(**kwargs))
				inputs[k] = tuple(new_labels)
				

		if self.args.past_index >= 0 and self._past is not None:
			inputs["mems"] = self._past

		return inputs

	def training_step(self, model, inputs):
		"""
		Perform a training step on a batch of inputs.

		Subclass and override to inject custom behavior.

		Args:
			model (:obj:`nn.Module`):
				The model to train.
			inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
				The inputs and targets of the model.

				The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
				argument :obj:`labels`. Check your model's documentation for all accepted arguments.

		Return:
			:obj:`torch.Tensor`: The tensor with training loss on this batch.
		"""

		model.train()
		inputs = self._prepare_inputs(inputs)

		loss = self.compute_loss(model, inputs)

		if self.args.n_gpu > 1:
			loss = loss.mean()

		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

		if self.use_apex:
			with amp.scale_loss(loss, self.optimizer) as scaled_loss:
				scaled_loss.backward()
		elif self.deepspeed:
			self.deepspeed.backward(loss)
		else:
			loss.backward()

		return loss.detach()
	
if __name__ == "__main__":
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	# TODO: Load checkpoint
	
	base_path = os.path.join("E:", os.sep, "Datasets", "common-voice-12")
	
	orthography = Orthography()
	orthography.tokenizer = model_args.tokenizer
	
	# create processor
	feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
		model_args.model_name_or_path, cache_dir=model_args.cache_dir
	)
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
		orthography.tokenizer,
		cache_dir=model_args.cache_dir,
		do_lower_case=orthography.do_lower_case,
		word_delimiter_token=orthography.word_delimiter_token
	)
	processor = Wav2Vec2Processor(feature_extractor, tokenizer)
	
	# Load dataset
	dataset = datasets.load_dataset('csv', data_files={'train': os.path.join(base_path, 'train.csv'), 'test': os.path.join(base_path, 'test.csv')})
	print(dataset)
	
	# create label maps
	cls_emotion_label_map = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
	cls_age_label_map = {'teens':0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5, 'seventies': 6}
	cls_gender_label_map = {'female': 0, 'male': 1}
	
	# Load model
	model = Wav2Vec2ForCTCnCLS.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		gradient_checkpointing=False,
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
			print("Example:", example[data_args.target_text_column])
			updated_text = " ".join(example[data_args.target_text_column].split()) # remove whitespaces
			updated_text = vocabulary_text_cleaner.sub("", updated_text)
			if updated_text != example[data_args.target_text_column]:
				example[data_args.target_text_column] = updated_text
		return example
	
	# remove samples without text
	dataset = dataset.filter(lambda example: example[data_args.target_text_column])
	
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
				if orthography.untransliterator is not None:
					logger.debug(f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
					logger.debug(f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

		wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)
		return {"acc": correct/total, "wer": wer, "correct": correct, "total": total, "strlen": len(ctc_label_str)}
		
	if model_args.freeze_feature_extractor:
		model.freeze_feature_extractor()
		
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