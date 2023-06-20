import torch
from typing import List
import numpy as np

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
		
		speaker_embedding_indices : List[int] = []
		speaker_embeddings = []
		label_features = []
		age_cls_labels = []
		gender_cls_labels = []		
		if self.audio_only is False:
			# labels shape is: inputs|gendercls|agecls|speaker embedding|speaker embedding length
			for i,feature in enumerate(features):
				speaker_embedding_indices.append(int(feature["labels"][-1]))
								
				# pad speaker embeddings with -1.0
				current_embeddings = feature["labels"][-1 - speaker_embedding_indices[-1]: -1]
				current_embedding_len = len(current_embeddings)				
				if (current_embedding_len > 1024*40):
					print("THIS SHOULD NOT HAPPEN. INCREASE 1024 padding size.")
				current_embeddings.extend([-1.0] * (1024*40 - current_embedding_len))
				speaker_embeddings.append(current_embeddings)
				
				label_features.append({"input_ids": [int(lf) for lf in feature["labels"][:-3 - speaker_embedding_indices[-1]]]})
				age_cls_labels.append(feature["labels"][-2 - speaker_embedding_indices[-1]])
				gender_cls_labels.append(feature["labels"][-3 - speaker_embedding_indices[-1]])
				
				#print("Collator labels: embedding index:", speaker_embedding_indices[-1])
				#print("Collator labels: embedding:", speaker_embeddings[-1])
				#print("Collator labels: features:", label_features[-1])
				#print("Collator labels: embedding index:", age_cls_labels[-1])
				#print("Collator labels: embedding index:", gender_cls_labels[-1])

		# speaker embedding is a list of list of float and has to be converted to a tensor
		print("se length:", len(speaker_embeddings), "se0 len:", len(speaker_embeddings[0]))
		speaker_embedding_tensors = [np.array(se) for se in speaker_embeddings]
		speaker_embedding_tensors = torch.tensor(speaker_embedding_tensors)

		print("set shape:", speaker_embedding_tensors.shape)
			
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

			batch["labels"] = (ctc_labels, torch.tensor(age_cls_labels), torch.tensor(gender_cls_labels), speaker_embedding_tensors)

		return batch