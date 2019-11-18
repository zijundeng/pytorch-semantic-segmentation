
import torch
from torch import nn
from torchvision import models

from ..utils import initialize_weights
from .seg_net import _DecoderBlock, SegNet

class SegNetBayes(SegNet):
	def __init__(self, num_classes, dropout_p=0.5, pretrained=True, num_samples=16, min_batch_size=4):
		"""
		@param num_samples: number of samples for the Monte-Carlo simulation,
			how many times to run with random dropout
		"""
		super().__init__(num_classes=num_classes, pretrained=pretrained)

		self.drop = nn.Dropout2d(p=dropout_p, inplace=False)
		self.num_samples = num_samples
		self.min_batch_size = min_batch_size

	def forward(self, x):
		enc1 = self.enc1(x)
		#print('enc1', enc1.shape)

		enc2 = self.enc2(enc1)
		#print('enc2', enc2.shape)

		enc3 = self.enc3(enc2)
		#print('enc3', enc3.shape)
		enc3 = self.drop(enc3)
		#print('enc3d', enc3.shape)

		enc4 = self.enc4(enc3)
		#print('enc4', enc4.shape)
		enc4 = self.drop(enc4)
		#print('enc4d', enc4.shape)

		enc5 = self.enc5(enc4)
		#print('enc5', enc5.shape)
		enc5 = self.drop(enc5)
		#print('enc5d', enc5.shape)

		dec5 = self.dec5(enc5)
		#print('dec5', dec5.shape)
		dec5 = self.drop(dec5)
		#print('dec5d', dec5.shape)

		dec4 = self.dec4(torch.cat([enc4, dec5], 1))
		#print('dec4', dec4.shape)
		dec4 = self.drop(dec4)
		#print('dec4d', dec4.shape)

		dec3 = self.dec3(torch.cat([enc3, dec4], 1))
		dec3 = self.drop(dec3)

		dec2 = self.dec2(torch.cat([enc2, dec3], 1))
		dec1 = self.dec1(torch.cat([enc1, dec2], 1))
		return dec1

	def forward_multisample(self, x, num_samples=None):
		# dropout must be active
		backup_train_mode = self.drop.training
		self.drop.train()

		softmax = torch.nn.Softmax2d()

		num_samples = num_samples if num_samples else self.num_samples

		results = [softmax(self.forward(x)).data.cpu() for i in range(num_samples)]

		preds = torch.stack(results).cuda()
		avg = torch.mean(preds, 0)
		var = torch.var(preds, 0)
		del preds

		# restore mode
		self.drop.train(backup_train_mode)

		return dict(
			mean = avg,
			var = var,
		)

	#def sample(self, x, num_samples, batch_size):
		#infer desired batch size from input shape
		#we will divide a num_samples into batches
		#num_frames = x.shape[0]
		#batch_size = max(num_frames, self.min_batch_size)

		#for sample_idx in range(num_samples):
			#pred =


		#for fr_idx in range(num_frames):
			#x_single = x[fr_idx:fr_idx+1, :, :, :]
			#self.sample(x_single, num_samples, batch_size)



