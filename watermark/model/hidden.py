import numpy as np
import torch
import torch.nn as nn

from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from options import HiDDenConfiguration
from vgg_loss import VGGLoss


class Hidden(nn.Module):
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser, tb_logger=None):
        super().__init__()
        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())
        self.vgg_loss = VGGLoss(3, 1, False).to(device) if configuration.use_vgg else None
        if self.vgg_loss is not None:
            self.vgg_loss.eval()
            for parameter in self.vgg_loss.parameters():
                parameter.requires_grad_(False)
        self.config = configuration
        self.device = device
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.cover_label = 1.0
        self.encoded_label = 0.0
        self.tb_logger = tb_logger

        if tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder.final_layer
            decoder_final = self.encoder_decoder.decoder.linear
            discriminator_final = self.discriminator.linear
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name("grads/encoder_out"))
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name("grads/decoder_out"))
            discriminator_final.weight.register_hook(tb_logger.grad_hook_by_name("grads/discriminator_out"))

    def _losses(self, images, messages, encoded_images, decoded_messages):
        batch_size = images.shape[0]
        cover_logits = self.discriminator(images)
        encoded_logits = self.discriminator(encoded_images)
        cover_targets = torch.full_like(cover_logits, self.cover_label)
        encoded_targets = torch.full_like(encoded_logits, self.encoded_label)
        generator_targets = torch.full_like(encoded_logits, self.cover_label)

        discriminator_cover_loss = self.bce_with_logits(cover_logits, cover_targets)
        discriminator_encoded_loss = self.bce_with_logits(encoded_logits, encoded_targets)
        adversarial_loss = self.bce_with_logits(encoded_logits, generator_targets)

        if self.vgg_loss is None:
            encoder_loss = self.mse_loss(encoded_images, images)
        else:
            with torch.no_grad():
                cover_features = self.vgg_loss(images)
            encoded_features = self.vgg_loss(encoded_images)
            encoder_loss = self.mse_loss(encoded_features, cover_features)

        decoder_loss = self.mse_loss(decoded_messages, messages)
        total_loss = (
            self.config.adversarial_loss * adversarial_loss
            + self.config.encoder_loss * encoder_loss
            + self.config.decoder_loss * decoder_loss
        )
        decoded_bits = decoded_messages.detach().round().clamp(0, 1)
        bitwise_error = (decoded_bits != messages.detach()).float().mean()

        losses = {
            "loss": total_loss,
            "encoder_mse": encoder_loss,
            "dec_mse": decoder_loss,
            "bitwise-error": bitwise_error,
            "adversarial_bce": adversarial_loss,
            "discr_cover_bce": discriminator_cover_loss,
            "discr_encod_bce": discriminator_encoded_loss,
        }
        return losses, (cover_logits, encoded_logits)

    @staticmethod
    def _loss_values(losses):
        return {name: value.detach().item() for name, value in losses.items()}

    def train_on_batch(self, batch: list):
        images, messages = batch
        images = images.to(self.device)
        messages = messages.to(self.device, dtype=images.dtype)
        self.train()

        encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

        self.optimizer_discrim.zero_grad(set_to_none=True)
        cover_logits = self.discriminator(images)
        encoded_logits = self.discriminator(encoded_images.detach())
        cover_targets = torch.full_like(cover_logits, self.cover_label)
        encoded_targets = torch.full_like(encoded_logits, self.encoded_label)
        discriminator_cover_loss = self.bce_with_logits(cover_logits, cover_targets)
        discriminator_encoded_loss = self.bce_with_logits(encoded_logits, encoded_targets)
        discriminator_loss = discriminator_cover_loss + discriminator_encoded_loss
        discriminator_loss.backward()
        self.optimizer_discrim.step()

        self.optimizer_enc_dec.zero_grad(set_to_none=True)
        discriminator_parameters = list(self.discriminator.parameters())
        for parameter in discriminator_parameters:
            parameter.requires_grad_(False)
        try:
            losses, _ = self._losses(images, messages, encoded_images, decoded_messages)
            losses["loss"].backward()
            self.optimizer_enc_dec.step()
        finally:
            for parameter in discriminator_parameters:
                parameter.requires_grad_(True)

        values = self._loss_values(losses)
        values["discr_cover_bce"] = discriminator_cover_loss.detach().item()
        values["discr_encod_bce"] = discriminator_encoded_loss.detach().item()
        return values, (encoded_images.detach(), noised_images.detach(), decoded_messages.detach())

    def validate_on_batch(self, batch: list):
        images, messages = batch
        images = images.to(self.device)
        messages = messages.to(self.device, dtype=images.dtype)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
                losses, _ = self._losses(images, messages, encoded_images, decoded_messages)
            if self.tb_logger is not None:
                self.tb_logger.add_tensor("weights/encoder_out", self.encoder_decoder.encoder.final_layer.weight)
                self.tb_logger.add_tensor("weights/decoder_out", self.encoder_decoder.decoder.linear.weight)
                self.tb_logger.add_tensor("weights/discrim_out", self.discriminator.linear.weight)
            return self._loss_values(losses), (encoded_images, noised_images, decoded_messages)
        finally:
            self.train(was_training)

    def to_string(self):
        return f"{self.encoder_decoder}\n{self.discriminator}"

    def to_stirng(self):
        return self.to_string()
