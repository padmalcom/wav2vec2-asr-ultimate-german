from transformers import Trainer
import torch

class CTCTrainer(Trainer):
	def training_step(self, model, inputs):
		model.train()
		inputs = self._prepare_inputs(inputs)

		loss = self.compute_loss(model, inputs)		

		if self.args.n_gpu > 1:
			loss = loss.mean()

		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

		loss.backward()

		return loss.detach()
