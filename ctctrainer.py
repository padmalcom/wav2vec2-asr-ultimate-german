from transformers import Trainer
import torch

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
		model.train()
		self.use_amp = True
		inputs = self._prepare_inputs(inputs)

		#loss = self.compute_loss(model, inputs)
		
		if self.use_amp:
			with autocast():
				loss = self.compute_loss(model, inputs)
		else:
			loss = self.compute_loss(model, inputs)		

		if self.args.n_gpu > 1:
			loss = loss.mean()

		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

		if self.use_amp:
			self.scaler.scale(loss).backward()
		elif self.use_apex:
			with amp.scale_loss(loss, self.optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()

		return loss.detach()
