import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from df.adaptive_dsp import SubbandNLMS
from df.config import config
from df.deepfilternet_defense import DefenseDfNet, init_model
from df.loss import DefenseEnhancedLoss
from df.modules import InstantaneousImpulseLimiter
from df.noise_classifier import DefenseNoiseClassifier


class TestDefenseUpgrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.use_defaults()

    def test_noise_classifier(self):
        classifier = DefenseNoiseClassifier(in_channels=32, num_classes=3)
        b, t, e = 4, 25, 32
        erb_feat = torch.randn(b, 1, t, e)

        logits, probs, confidence = classifier(erb_feat)

        self.assertEqual(logits.shape, (b, 3))
        self.assertEqual(probs.shape, (b, 3))
        self.assertEqual(confidence.shape, (b, 1))

        # Check valid probability distribution
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(b), atol=1e-5, rtol=1e-5)

        # Check backward gradient flow
        loss = logits.sum()
        loss.backward()
        for p in classifier.parameters():
            self.assertIsNotNone(p.grad)

    def test_impulse_limiter(self):
        limiter = InstantaneousImpulseLimiter(
            crest_thresh_db=12.0, energy_thresh_rel=5.0, max_atten_db=20.0
        )
        b, c, t, f = 1, 1, 30, 481
        spec = torch.randn(b, c, t, f, 2) * 0.1
        feat_erb = torch.randn(b, c, t, 32) * 0.1

        # Inject extreme acoustic shockwave at frame 15
        spec[:, :, 15, :, :] *= 50.0
        feat_erb[:, :, 15, :] *= 50.0

        spec_lim, erb_lim, impulse_mask = limiter(spec, feat_erb)

        self.assertEqual(spec_lim.shape, spec.shape)
        self.assertEqual(erb_lim.shape, feat_erb.shape)
        self.assertEqual(impulse_mask.shape, (b, t, 1))

        # Verify frame 15 was flagged as impulse
        self.assertEqual(impulse_mask[0, 15, 0].item(), 1.0)
        # Verify other quiet frames were not flagged
        self.assertEqual(impulse_mask[0, 0, 0].item(), 0.0)

        # Verify shock energy was attenuated
        shock_in_energy = (spec[:, :, 15, :, :] ** 2).sum()
        shock_out_energy = (spec_lim[:, :, 15, :, :] ** 2).sum()
        self.assertLess(shock_out_energy, shock_in_energy * 0.1)

    def test_subband_nlms(self):
        nlms = SubbandNLMS(num_freqs=481, filter_order=2, mu_init=0.2)
        b, t, f = 1, 40, 481

        # Synthesize correlated stationary noise
        ref_noise = torch.randn(b, 1, t, f, 2)
        coupling = 0.8
        primary_noise = ref_noise * coupling
        speech = torch.randn(b, 1, t, f, 2) * 0.05
        primary_spec = speech + primary_noise

        probs = torch.tensor([[0.8, 0.1, 0.1]])  # Stationary noise regime
        err_spec, noise_est = nlms(primary_spec, ref_noise, noise_probs=probs)

        self.assertEqual(err_spec.shape, primary_spec.shape)
        self.assertEqual(noise_est.shape, primary_spec.shape)

        # Later frames should show cancellation of the correlated noise
        initial_error = torch.mean((err_spec[:, :, 2:5] - speech[:, :, 2:5]) ** 2)
        adapted_error = torch.mean((err_spec[:, :, -5:] - speech[:, :, -5:]) ** 2)
        self.assertLess(adapted_error.item(), initial_error.item())

    def test_defense_dfnet_end_to_end(self):
        model = init_model()
        self.assertIsInstance(model, DefenseDfNet)

        b, t = 2, 8
        f = model.freq_bins
        e = model.erb_bins
        f_df = model.nb_df

        spec = torch.randn(b, 1, t, f, 2)
        feat_erb = torch.randn(b, 1, t, e).abs()
        feat_spec = torch.randn(b, 1, t, f_df, 2)
        ref_spec = torch.randn(b, 1, t, f, 2)

        out = model(spec, feat_erb, feat_spec, ref_spec=ref_spec)
        fused_spec, mask, lsnr, df_coefs, noise_probs, impulse_mask, fusion_weights = out

        # Verify expected shapes
        self.assertEqual(fused_spec.shape, (b, 1, t, f, 2))
        self.assertEqual(mask.shape, (b, 1, t, e))
        self.assertEqual(lsnr.shape, (b, t, 1))
        self.assertEqual(df_coefs.shape, (b, model.df_order, t, f_df, 2))
        self.assertEqual(noise_probs.shape, (b, 3))
        self.assertEqual(impulse_mask.shape, (b, t, 1))
        self.assertEqual(fusion_weights.shape, (b, t, 2))

        # Check backward pass
        loss = fused_spec.sum() + mask.sum() + lsnr.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and "df_fc_a" not in name:
                self.assertIsNotNone(param.grad, f"Parameter {name} missing gradient")

    def test_defense_loss(self):
        loss_fn = DefenseEnhancedLoss(
            factor_spec=1.0, factor_mask=1.0, factor_cls=0.1, factor_impulse=2.0
        )
        b, t, f, e = 2, 6, 481, 32

        clean_spec = torch.randn(b, 1, t, f, 2)
        enh_spec = clean_spec + torch.randn(b, 1, t, f, 2) * 0.1
        mask_pred = torch.sigmoid(torch.randn(b, 1, t, e))
        mask_target = torch.sigmoid(torch.randn(b, 1, t, e))
        noise_logits = torch.randn(b, 3)
        noise_labels = torch.tensor([0, 2])
        impulse_mask = torch.zeros(b, t, 1)
        impulse_mask[1, 3, 0] = 1.0

        loss, metrics = loss_fn(
            clean_spec=clean_spec,
            enh_spec=enh_spec,
            mask_pred=mask_pred,
            mask_target=mask_target,
            noise_logits=noise_logits,
            noise_labels=noise_labels,
            impulse_mask=impulse_mask,
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_spec", metrics)
        self.assertIn("loss_mask", metrics)
        self.assertIn("loss_cls", metrics)
        self.assertIn("loss_impulse", metrics)
        self.assertIn("loss_total", metrics)


if __name__ == "__main__":
    unittest.main()
