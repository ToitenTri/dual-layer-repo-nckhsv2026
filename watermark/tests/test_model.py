import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.decoder import Decoder
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.hidden import Hidden
from noise_argparser import NoiseArgParser
from noise_layer.crop import Crop, get_random_rectangle_inside
from noise_layer.identity import Identity
from noise_layer.jpeg_compression import JpegCompression
from noise_layer.noiser import Noiser
from options import HiDDenConfiguration
from ultils import save_images


def small_config():
    return HiDDenConfiguration(
        H=16,
        W=16,
        message_length=8,
        encoder_blocks=1,
        encoder_channels=8,
        decoder_blocks=1,
        decoder_channels=8,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=1,
        discriminator_channels=8,
        decoder_loss=1.0,
        encoder_loss=0.7,
        adversarial_loss=0.001,
    )


class ModelTests(unittest.TestCase):
    def test_encoder_decoder_preserves_batch_dimension(self):
        config = small_config()
        model = EncoderDecoder(config, Noiser(device=torch.device("cpu")))
        for batch_size in (1, 2):
            image = torch.randn(batch_size, 3, 16, 16)
            message = torch.randint(0, 2, (batch_size, 8)).float()
            encoded, noised, decoded = model(image, message)
            self.assertEqual(encoded.shape, image.shape)
            self.assertEqual(noised.shape, image.shape)
            self.assertEqual(decoded.shape, message.shape)

    def test_decoder_and_discriminator_keep_batch_dimension(self):
        config = small_config()
        image = torch.randn(1, 3, 16, 16)
        self.assertEqual(Decoder(config)(image).shape, (1, 8))
        self.assertEqual(Discriminator(config)(image).shape, (1, 1))

    def test_hidden_train_and_validation_step(self):
        model = Hidden(small_config(), torch.device("cpu"), Noiser(device=torch.device("cpu")))
        image = torch.randn(1, 3, 16, 16)
        message = torch.randint(0, 2, (1, 8)).float()

        losses, outputs = model.train_on_batch([image, message])
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        self.assertEqual(outputs[2].shape, message.shape)

        validation_losses, validation_outputs = model.validate_on_batch([image, message])
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in validation_losses.values()))
        self.assertEqual(validation_outputs[2].shape, message.shape)
        self.assertTrue(model.training)

    def test_identity_noise_layer_is_callable(self):
        pair = [torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)]
        self.assertIs(Identity()(pair), pair)

    def test_crop_samples_width_from_full_range_and_keeps_pair_aligned(self):
        image = torch.randn(1, 3, 20, 20)
        sampled_ranges = []

        def sample(low, high):
            sampled_ranges.append((low, high))
            return (low + high) / 2

        with patch("noise_layer.crop.random_float", side_effect=sample):
            rectangle = get_random_rectangle_inside(image, (0.5, 0.9), (0.4, 0.8))
            result = Crop((0.5, 0.9), (0.4, 0.8))([image.clone(), image.clone()])
        self.assertEqual(sampled_ranges, [(0.5, 0.9), (0.4, 0.8)] * 2)

        self.assertEqual(result[0].shape, result[1].shape)
        self.assertEqual(result[0].shape[2:], (rectangle[1] - rectangle[0], rectangle[3] - rectangle[2]))

    def test_dropout_keeps_noise_pair_contract(self):
        pair = [torch.ones(2, 3, 8, 8), torch.zeros(2, 3, 8, 8)]
        output = Noiser([{"type": "dropout", "keep_ratio_range": (0.5, 0.9)}])(pair)
        self.assertIsInstance(output, list)
        self.assertEqual(output[0].shape, pair[0].shape)
        self.assertIs(output[1], pair[1])

    def test_noise_argument_parser_accepts_supported_layers(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--noise", nargs="*", action=NoiseArgParser)
        parsed = parser.parse_args(["--noise", "crop((0.5,0.9),(0.4,0.8)) + resize(0.5) + jpeg()"])
        self.assertEqual([layer["type"] for layer in parsed.noise], ["crop", "resize", "jpeg_compression"])

    def test_jpeg_noise_preserves_non_multiple_of_eight_dimensions(self):
        image = torch.randn(1, 3, 15, 19)
        cover = torch.randn_like(image)
        output = JpegCompression(torch.device("cpu"))([image, cover])
        self.assertEqual(output[0].shape, image.shape)
        self.assertTrue(torch.isfinite(output[0]).all())

    def test_jpeg_noise_supports_training_backward(self):
        model = Hidden(
            small_config(),
            torch.device("cpu"),
            Noiser([{"type": "jpeg_compression"}], torch.device("cpu")),
        )
        image = torch.randn(1, 3, 16, 16)
        message = torch.randint(0, 2, (1, 8)).float()
        losses, _ = model.train_on_batch([image, message])
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))

    def test_save_images_uses_current_torchvision_signature(self):
        image = torch.randn(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as folder:
            save_images(image, image, 1, folder, resize_to=(32, 32))
            self.assertTrue((Path(folder) / "epoch-1.png").is_file())


if __name__ == "__main__":
    unittest.main()
