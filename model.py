import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

class Wav2Vec2ForCTCnCLS(Wav2Vec2PreTrainedModel):

	def __init__(self, config, age_cls_len, gender_cls_len, age_cls_weights, gender_cls_weights, alpha=0.01):
		super().__init__(config)
		self.wav2vec2 = Wav2Vec2Model(config)
		self.dropout = nn.Dropout(config.final_dropout)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
		self.age_cls_head = nn.Linear(config.hidden_size, age_cls_len)
		self.gender_cls_head = nn.Linear(config.hidden_size, gender_cls_len)
		self.init_weights()
		self.age_cls_weights = age_cls_weights
		self.gender_cls_weights = gender_cls_weights
		self.alpha = alpha
		
		mel_n_channels = 40
		model_num_layers = 3
		model_embedding_size = 256
		
		# speaker embedding
		self.lstm = nn.LSTM(
			input_size=mel_n_channels,
			hidden_size=config.hidden_size, 
			num_layers=model_num_layers, 
			batch_first=True
		)
		
		self.linear = nn.Linear(
			in_features=config.hidden_size, 
			out_features=model_embedding_size
		)
		self.relu = torch.nn.ReLU()
		
		self.similarity_weight = nn.Parameter(torch.tensor([10.]))
		self.similarity_bias = nn.Parameter(torch.tensor([-5.]))
		
		self.embedding_loss_fn = nn.CrossEntropyLoss()
		
	# used for speaker embedding
	def do_gradient_ops(self):
		# Gradient scale
		self.similarity_weight.grad *= 0.01
		self.similarity_bias.grad *= 0.01
			
		# Gradient clipping
		clip_grad_norm_(self.parameters(), 3, norm_type=2)

	def freeze_feature_extractor(self):
		self.wav2vec2.feature_extractor._freeze_parameters()

	def _ctc_loss(self, logits, labels, input_values, attention_mask=None):
		loss = None
		if labels is not None:

			# retrieve loss input_lengths from attention_mask
			attention_mask = (
				attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
			)
			input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

			# assuming that padded tokens are filled with -100
			# when not being attended to
			labels_mask = labels >= 0
			target_lengths = labels_mask.sum(-1)
			flattened_targets = labels.masked_select(labels_mask)

			log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

			with torch.backends.cudnn.flags(enabled=False):
				loss = F.ctc_loss(
					log_probs,
					flattened_targets,
					input_lengths,
					target_lengths,
					blank=self.config.pad_token_id,
					reduction=self.config.ctc_loss_reduction,
					zero_infinity=self.config.ctc_zero_infinity,
					)

		return loss

	# use this function for all classification tasks
	def _cls_loss(self, logits, cls_labels, cls_weights): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
		loss = None
		if cls_labels is not None:
			loss = F.cross_entropy(logits, cls_labels.to(logits.device), weight=torch.tensor(cls_weights, device=logits.device, dtype=torch.float))
		return loss
	
	# used for speaker embedding	
	def _speaker_embedding_loss(self, embeds):
		"""
		Computes the softmax loss according the section 2.1 of GE2E.
		
		:param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
		utterances_per_speaker, embedding_size)
		:return: the loss and the EER for this batch of embeddings.
		"""
		speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
		
		# Loss
		sim_matrix = self.similarity_matrix(embeds)
		sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
										 speakers_per_batch))
		ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
		target = torch.from_numpy(ground_truth).long().to(self.loss_device)
		loss = self.loss_fn(sim_matrix, target)
		
		# EER (not backpropagated)
		with torch.no_grad():
			inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
			labels = np.array([inv_argmax(i) for i in ground_truth])
			preds = sim_matrix.detach().cpu().numpy()

			# Snippet from https://yangcha.github.io/EER-ROC/
			fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())		   
			eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
			
		return loss, eer
	

	# used for speaker embeddings
	def similarity_matrix(self, embeds):
		"""
		Computes the similarity matrix according the section 2.1 of GE2E.

		:param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
		utterances_per_speaker, embedding_size)
		:return: the similarity matrix as a tensor of shape (speakers_per_batch,
		utterances_per_speaker, speakers_per_batch)
		"""
		speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
		
		# Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
		centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
		centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

		# Exclusive centroids (1 per utterance)
		centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
		centroids_excl /= (utterances_per_speaker - 1)
		centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

		# Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
		# product of these vectors (which is just an element-wise multiplication reduced by a sum).
		# We vectorize the computation for efficiency.
		sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
								 speakers_per_batch).to(self.loss_device)
		mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
		for j in range(speakers_per_batch):
			mask = np.where(mask_matrix[j])[0]
			sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
			sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
				
		sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
		return sim_matrix

	def forward(
		self,
		input_values,
		attention_mask=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		labels=None, # tuple: (ctc_labels, age_cls_labels, gender_cls_labels), shape=(batch_size, target_length)
		hidden_init=None
		):
		
		print("input type:", type(input_values), "shape:", input_values.shape, "Input in forward is:", input_values)
		
		print("labels type:", type(labels), "Labels are:", labels)

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0] # this is the last layer's hidden states
		hidden_states = self.dropout(hidden_states)

		logits_ctc = self.lm_head(hidden_states)
		logits_age_cls = self.age_cls_head(torch.mean(hidden_states, dim=1))
		logits_gender_cls = self.gender_cls_head(torch.mean(hidden_states, dim=1))
		
		#speaker embedding
		print("input values:", input_values.shape)
		utterances = hidden_states
		out, (hidden, cell) = self.lstm(utterances, hidden_init)
		
		# We take only the hidden state of the last layer
		embeds_raw = self.relu(self.linear(hidden[-1]))
		
		# L2-normalize it
		embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)	
		
		loss = None
		if labels is not None:
			#print("labels in forward:", "label1 (age):", labels[1], "label2 (gender):", labels[2])
			loss_ctc = self._ctc_loss(logits_ctc, labels[0], input_values, attention_mask)
			loss_age_cls = self._cls_loss(logits_age_cls, labels[1], self.age_cls_weights)
			loss_gender_cls = self._cls_loss(logits_gender_cls, labels[2], self.gender_cls_weights)
			#loss_speaker_embedding = self._speaker_embedding_loss(embeds)
			loss_speaker_embedding = 0
			print("Loss speaker embedding:", loss_speaker_embedding, "loss age cls:", loss_age_cls, "loss gender cls:", loss_gender_cls, "loss ctc:", loss_ctc)
			loss = loss_age_cls + loss_gender_cls + self.alpha * loss_ctc + loss_speaker_embedding
		
		return CausalLMOutput(
			#loss=loss, logits=(logits_ctc, logits_age_cls, logits_gender_cls, embeds), hidden_states=outputs.hidden_states, attentions=outputs.attentions
			loss=loss, logits=(logits_ctc, logits_age_cls, logits_gender_cls, embeds), hidden_states=outputs.hidden_states, attentions=outputs.attentions
		)

		
