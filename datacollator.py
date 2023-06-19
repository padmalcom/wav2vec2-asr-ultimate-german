import torch
from typing import List

class DataCollatorCTCWithPadding:
	def __init__(self, processor, padding, audio_only=False):
		self.processor = processor
		self.padding = padding
		self.max_length = None
		self.max_length_labels = None
		self.pad_to_multiple_of = None
		self.pad_to_multiple_of_labels = None
		self.audio_only = audio_only

	def __call__(self, features):
		# split inputs and labels since they have to be of different lenghts and need
		# different padding methods
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		
		#print("Batch in collator:", features[0]["labels"])
		#input_mels = [{"mels": feature["mel"]} for feature in features]
		#print("mel in collator:", mels.shape)
		if self.audio_only is False:
			speaker_embedding_indices : List[int] = []
			speaker_embeddings = []
			label_features = []
			age_cls_labels = []
			gender_cls_labels = []
			# labels shape is: inputs|gendercls|agecls|speaker embedding|speaker embedding length
			for i,feature in enumerate(features):
				speaker_embedding_indices.append(int(feature["labels"][-1]))
				speaker_embeddings.append(feature["labels"][-1 - speaker_embedding_indices[-1]: -1])
				label_features.append({"input_ids": [int(lf) for lf in feature["labels"][:-3 - speaker_embedding_indices[-1]]]})
				age_cls_labels.append(feature["labels"][-2 - speaker_embedding_indices[-1]])
				gender_cls_labels.append(feature["labels"][-3 - speaker_embedding_indices[-1]])
				
				#print("Collator labels: embedding index:", speaker_embedding_indices[-1])
				#print("Collator labels: embedding:", speaker_embeddings[-1])
				#print("Collator labels: features:", label_features[-1])
				#print("Collator labels: embedding index:", age_cls_labels[-1])
				#print("Collator labels: embedding index:", gender_cls_labels[-1])
				
			
		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)
		print("0")
		if self.audio_only is False:
			print("1")
			with self.processor.as_target_processor():
				labels_batch = self.processor.pad(
					label_features,
					padding=self.padding,
					max_length=self.max_length_labels,
					pad_to_multiple_of=self.pad_to_multiple_of_labels,
					return_tensors="pt",
				)
			print("2")
			# replace padding with -100 to ignore loss correctly
			ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
			print("2.5")
			batch["labels"] = (ctc_labels, torch.tensor(age_cls_labels), torch.tensor(gender_cls_labels)) #, torch.tensor(speaker_embeddings)
			print("3")
		print("4")
		return batch