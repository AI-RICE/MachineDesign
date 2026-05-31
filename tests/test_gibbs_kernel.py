"""Unit tests for the Gibbs / anisotropic Paciorek-Schervish kernel sampler.

Maps to PLAN_gibbs_prior.md phase 1-4 validation gates (tests 1-21). Phase 5/6
gates are integration tests run elsewhere (CLI smoke test + 60k-step
confirmation training).

Run with:
    cd applications/ReluctanceDrive/MachineDesign
    python -m pytest tests/test_gibbs_kernel.py -v
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from machine_design.pfn.gibbs_prior_sampler import (
    GibbsPriorConfig,
    GibbsPriorSampler,
    log_ell_field,
    paciorek_schervish_matern52,
)
from machine_design.pfn.gp_prior_sampler import GPPriorConfig, GPPriorSampler


# ============================================================
# Phase 1 -- Length-scale field
# ============================================================
class TestLengthScaleField:
    def test_01_shape(self):
        a = np.zeros(7); b = np.ones(7)
        x = np.random.default_rng(0).uniform(-1, 1, (50, 7))
        assert log_ell_field(a, b, x).shape == (50, 7)

    def test_02_stationary_recovery_b_zero(self):
        a = np.array([0.5, -0.3, 1.0]); b = np.zeros(3)
        x = np.random.default_rng(1).uniform(-1, 1, (20, 3))
        log_ell = log_ell_field(a, b, x)
        # Each column should be constant = a_d
        for d in range(3):
            assert np.allclose(log_ell[:, d], a[d])

    def test_03_sign_convention_b_positive(self):
        a = np.zeros(2); b = np.array([1.0, 2.0])
        x_pos = np.full((1, 2), +1.0); x_neg = np.full((1, 2), -1.0)
        assert (log_ell_field(a, b, x_pos) > log_ell_field(a, b, x_neg)).all()

    def test_04_magnitude(self):
        a = np.zeros(1); b = np.ones(1)
        x = np.array([[+1.0]])
        log_ell = log_ell_field(a, b, x)
        ell = np.exp(log_ell)
        assert math.isclose(float(ell.item()), math.e, rel_tol=1e-12)

    def test_05_positivity_bounded(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-3, 3, 7); b = rng.uniform(-3, 3, 7)
        x = rng.uniform(-1, 1, (100, 7))
        ell = np.exp(log_ell_field(a, b, x))
        assert (ell > 0).all() and np.isfinite(ell).all()

    def test_06_vectorised_matches_loop(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 4); b = rng.normal(0, 1, 4)
        x = rng.uniform(-1, 1, (20, 4))
        vec = log_ell_field(a, b, x)
        loop = np.empty_like(vec)
        for i in range(x.shape[0]):
            for d in range(x.shape[1]):
                loop[i, d] = a[d] + b[d] * x[i, d]
        assert np.allclose(vec, loop, atol=1e-15)


# ============================================================
# Phase 2 -- Paciorek-Schervish kernel matrix
# ============================================================
class TestGibbsKernel:
    def _draw(self, rng, N=12, D=4, b_scale=1.0):
        a = rng.normal(0, 1.0, D)
        b = rng.normal(0, b_scale, D)
        X = rng.uniform(-1, 1, (N, D))
        log_ell = log_ell_field(a, b, X)
        outputscale = float(np.exp(rng.normal(0, 0.5)))
        return X, log_ell, outputscale, a, b

    def test_07_symmetry(self):
        rng = np.random.default_rng(10)
        X, le, sf, _, _ = self._draw(rng)
        K = paciorek_schervish_matern52(X, X, le, le, sf)
        assert np.allclose(K, K.T, atol=1e-10)

    def test_08_diagonal_value(self):
        rng = np.random.default_rng(11)
        X, le, sf, _, _ = self._draw(rng)
        K = paciorek_schervish_matern52(X, X, le, le, sf)
        assert np.allclose(np.diag(K), sf, atol=1e-12)

    def test_09_stationary_recovery_matches_matern_arad(self):
        """With b = 0, Gibbs-Matern52 must equal the stationary Matern-5/2 kernel
        from gp_prior_sampler with ell = exp(a)."""
        from machine_design.pfn.gp_prior_sampler import GPPriorSampler
        rng = np.random.default_rng(12)
        D = 5; N = 15
        a = rng.normal(0, 0.5, D); b = np.zeros(D)
        X = rng.uniform(-1, 1, (N, D))
        log_ell = log_ell_field(a, b, X)
        outputscale = float(np.exp(rng.normal(0, 0.3)))
        K_gibbs = paciorek_schervish_matern52(X, X, log_ell, log_ell, outputscale)
        # gp_prior_sampler._matern signature: (X, ell, outputscale, nu).
        bounds = np.stack([np.zeros(D), np.ones(D)])
        ref = GPPriorSampler(input_dim=D, bounds=bounds)
        K_ref = ref._matern(X, ell=np.exp(a), outputscale=outputscale, nu=2.5)
        assert np.allclose(K_gibbs, K_ref, atol=1e-10), (
            f"max abs diff = {np.abs(K_gibbs - K_ref).max():.3e}")

    def test_10_stationary_zero_b_invariant_to_x(self):
        """With b = 0 the kernel must be translation-invariant in x_unit."""
        rng = np.random.default_rng(13)
        D = 3; N = 8
        a = rng.normal(0, 0.5, D); b = np.zeros(D)
        X1 = rng.uniform(-1, 1, (N, D))
        X2 = X1 + 0.3  # translation
        le1 = log_ell_field(a, b, X1); le2 = log_ell_field(a, b, X2)
        K1 = paciorek_schervish_matern52(X1, X1, le1, le1, 1.0)
        K2 = paciorek_schervish_matern52(X2, X2, le2, le2, 1.0)
        assert np.allclose(K1, K2, atol=1e-12)

    def test_11_positive_definite(self):
        rng = np.random.default_rng(14)
        for _ in range(30):
            X, le, sf, _, _ = self._draw(rng, N=20, b_scale=1.5)
            K = paciorek_schervish_matern52(X, X, le, le, sf)
            eig = np.linalg.eigvalsh(K + 1e-6 * np.eye(K.shape[0]))
            assert eig.min() >= -1e-8, f"non-PSD: min eig = {eig.min():.3e}"

    def test_12_norm_factor_in_unit_interval(self):
        """The normalising factor sqrt(2 l l' / (l^2 + l'^2)) is in (0, 1]."""
        rng = np.random.default_rng(15)
        D = 4; N = 10
        a = rng.normal(0, 1, D); b = rng.normal(0, 1.5, D)
        X = rng.uniform(-1, 1, (N, D))
        log_ell = log_ell_field(a, b, X)
        K = paciorek_schervish_matern52(X, X, log_ell, log_ell, outputscale=1.0)
        # K_ii = 1; K_ij = norm * matern. matern <= 1 (Matern-5/2 at any distance).
        # And norm <= 1. So K_ij in [0, 1]. Strict for off-diagonal in general.
        assert (K >= -1e-12).all()
        assert (K <= 1.0 + 1e-10).all()

    def test_13_cauchy_schwarz(self):
        rng = np.random.default_rng(16)
        X, le, sf, _, _ = self._draw(rng, N=15)
        K = paciorek_schervish_matern52(X, X, le, le, sf)
        for i in range(K.shape[0]):
            for j in range(K.shape[0]):
                ub = math.sqrt(K[i, i] * K[j, j]) + 1e-10
                assert abs(K[i, j]) <= ub, f"({i},{j}) violates Cauchy-Schwarz"

    def test_14_scalar_loop_reference(self):
        """5-line scalar loop reference vs vectorised. Catches indexing bugs."""
        rng = np.random.default_rng(17)
        D = 3; N = 6
        a = rng.normal(0, 0.5, D); b = rng.normal(0, 0.8, D)
        X = rng.uniform(-1, 1, (N, D))
        log_ell = log_ell_field(a, b, X)
        ell_sq = np.exp(2 * log_ell)
        outputscale = 1.3
        K_loop = np.zeros((N, N))
        SQRT5 = math.sqrt(5.0)
        for i in range(N):
            for j in range(N):
                e_i = ell_sq[i]; e_j = ell_sq[j]
                den = e_i + e_j                                # (D,)
                norm = np.prod(np.sqrt(2 * np.sqrt(e_i * e_j) / den))
                d2 = 2 * ((X[i] - X[j]) ** 2 / den).sum()
                d = math.sqrt(max(d2, 0.0))
                K_loop[i, j] = outputscale * norm * (1 + SQRT5 * d + 5 / 3 * d2) * math.exp(-SQRT5 * d)
        K_vec = paciorek_schervish_matern52(X, X, log_ell, log_ell, outputscale)
        assert np.allclose(K_vec, K_loop, atol=1e-12), (
            f"vectorised vs loop max diff = {np.abs(K_vec - K_loop).max():.3e}")


# ============================================================
# Phase 3 -- Sampling y
# ============================================================
class TestSampling:
    def test_15_cholesky_succeeds(self):
        rng = np.random.default_rng(20)
        sampler = GibbsPriorSampler(
            input_dim=7,
            bounds=np.stack([np.zeros(7), np.ones(7)]),
            cfg=GibbsPriorConfig(log_b_std=1.5),
        )
        # 50 random configs; every sample() call internally runs Cholesky.
        for _ in range(50):
            t = sampler.sample(rng, n_context=40, n_target=24, normalise=False)
            assert np.isfinite(t.y_context).all()
            assert np.isfinite(t.y_target).all()

    def test_16_empirical_covariance_matches_K(self):
        """5000 samples on a fixed K -> empirical cov(Y) matches K + noise^2 I."""
        rng = np.random.default_rng(21)
        D = 3; N = 8
        a = np.array([0.2, -0.3, 0.1]); b = np.array([0.5, -0.2, 0.3])
        X = rng.uniform(-1, 1, (N, D))
        log_ell = log_ell_field(a, b, X)
        outputscale = 0.7; noise = 0.05
        K = paciorek_schervish_matern52(X, X, log_ell, log_ell, outputscale)
        K_full = K + noise * np.eye(N)
        L = np.linalg.cholesky(K_full + 1e-9 * np.eye(N))
        n_samples = 5000
        Z = rng.standard_normal((n_samples, N))
        Y = Z @ L.T
        cov_emp = (Y.T @ Y) / n_samples
        max_diff = np.abs(cov_emp - K_full).max()
        # Sample-cov standard error ~ 1/sqrt(n_samples) ~ 1.4% for n=5000.
        assert max_diff < 0.05, f"empirical cov off by {max_diff:.4f}"

    def test_17_marginal_variance(self):
        """For many task draws at the same fixed x, var(y(x)) ~ outputscale + noise."""
        rng = np.random.default_rng(22)
        D = 4
        bounds = np.stack([np.zeros(D), np.ones(D)])
        # Fix outputscale and noise; only ls field varies per task. Use degenerate
        # GibbsPriorConfig by manually overriding what `sample` draws -- easiest:
        # subclass GibbsPriorSampler with deterministic outputscale/noise.
        outputscale_fixed = 0.5; noise_fixed = 0.01

        class _S(GibbsPriorSampler):
            def sample(self, rng, n_context, n_target, normalise=False):
                from machine_design.pfn.gibbs_prior_sampler import (
                    _stable_chol, log_ell_field, paciorek_schervish_matern52,
                )
                from machine_design.pfn.prior_sampler import PFNTask
                N = n_context + n_target; D = self.input_dim
                a = rng.normal(0, self.cfg.log_a_std, D)
                b = rng.normal(0, self.cfg.log_b_std, D)
                X_unit01 = rng.uniform(0, 1, (N, D)); X_unit = 2 * X_unit01 - 1
                log_ell = log_ell_field(a, b, X_unit)
                K = paciorek_schervish_matern52(X_unit, X_unit, log_ell, log_ell, outputscale_fixed)
                K = K + noise_fixed * np.eye(N)
                L = _stable_chol(K)
                y = L @ rng.standard_normal(N)
                X_raw = self.bounds[0] + X_unit01 * (self.bounds[1] - self.bounds[0])
                return PFNTask(X_raw[:n_context], y[:n_context], X_raw[n_context:], y[n_context:], "Gibbs")

        s = _S(input_dim=D, bounds=bounds, cfg=GibbsPriorConfig(log_b_std=0.5))
        ys = []
        for _ in range(2000):
            t = s.sample(rng, n_context=1, n_target=0, normalise=False)
            ys.append(t.y_context[0])
        var_emp = float(np.var(ys))
        # Expected var(y(x)) = outputscale + noise = 0.51. SE for n=2000 is ~3%.
        assert abs(var_emp - (outputscale_fixed + noise_fixed)) < 0.05, (
            f"marginal var={var_emp:.4f}, expected ~{outputscale_fixed + noise_fixed:.4f}")

    def test_18_stationary_reduction_matches_gp_sampler(self):
        """With b=0 the Gibbs sampler should produce 2-point covariances matching
        the wide stationary GPPriorSampler. Compared on the same x-pair via many
        task draws."""
        rng = np.random.default_rng(23)
        D = 4
        bounds = np.stack([np.zeros(D), np.ones(D)])
        # Match wide GP config: log_ls_std=1.4, log_outputscale_std=0.7, nu=2.5.
        gibbs_cfg = GibbsPriorConfig(log_a_std=1.4, log_b_std=0.0, log_outputscale_std=0.7,
                                     log_noise_min=-10, log_noise_max=-10)  # ~0 noise
        gp_cfg = GPPriorConfig(log_ls_std=1.4, log_outputscale_std=0.7,
                               log_noise_min=-10, log_noise_max=-10, nu_choices=(2.5,))
        gibbs = GibbsPriorSampler(D, bounds, gibbs_cfg)
        gp = GPPriorSampler(D, bounds, gp_cfg)
        # Compare marginal variance across many tasks (should match).
        n_tasks = 1000; y_g, y_p = [], []
        for _ in range(n_tasks):
            t = gibbs.sample(rng, n_context=1, n_target=0, normalise=False); y_g.append(t.y_context[0])
            t = gp.sample(rng, n_context=1, n_target=0, normalise=False); y_p.append(t.y_context[0])
        v_g, v_p = float(np.var(y_g)), float(np.var(y_p))
        # log-Normal outputscale has heavy tail; tolerate 25% relative diff.
        assert abs(v_g - v_p) / max(v_g, v_p) < 0.25, (
            f"Gibbs (b=0) marginal var {v_g:.3f} vs GP {v_p:.3f}")


# ============================================================
# Phase 4 -- Sampler class
# ============================================================
class TestSamplerAPI:
    def _make(self, D=5, log_b_std=1.0):
        bounds = np.stack([np.zeros(D), np.ones(D)])
        return GibbsPriorSampler(D, bounds, GibbsPriorConfig(log_b_std=log_b_std))

    def test_19_interface_parity(self):
        s = self._make()
        rng = np.random.default_rng(30)
        t = s.sample(rng, n_context=20, n_target=5, normalise=True)
        assert t.x_context.shape == (20, 5)
        assert t.y_context.shape == (20,)
        assert t.x_target.shape == (5, 5)
        assert t.y_target.shape == (5,)
        assert t.granularity == "Gibbs"

    def test_20_no_target_leak(self):
        """CRUCIAL regression test for the leak that motivated the rewrite.
        With normalise=True (context-only), corr(z_target, -sum z_context)
        should be near zero. If the leak returned, this would be ~ 1.0."""
        s = self._make(log_b_std=0.5)
        rng = np.random.default_rng(31)
        target_z, neg_sum_ctx = [], []
        for _ in range(150):
            t = s.sample(rng, n_context=40, n_target=1, normalise=True)
            target_z.append(t.y_target[0])
            neg_sum_ctx.append(-float(t.y_context.sum()))
        corr = float(np.corrcoef(target_z, neg_sum_ctx)[0, 1])
        assert abs(corr) < 0.5, f"Possible target leak: corr={corr:.3f}"

    def test_21_determinism(self):
        s = self._make()
        rng1 = np.random.default_rng(42); rng2 = np.random.default_rng(42)
        t1 = s.sample(rng1, n_context=10, n_target=3, normalise=False)
        t2 = s.sample(rng2, n_context=10, n_target=3, normalise=False)
        assert np.array_equal(t1.x_context, t2.x_context)
        assert np.array_equal(t1.y_context, t2.y_context)
        assert np.array_equal(t1.y_target, t2.y_target)


# Phase 5/6 gates are integration tests run elsewhere:
#   22. CPU smoke test of  python -m machine_design.pfn.train --prior gibbs --steps 100
#   23. Checkpoint round-trip via load_checkpoint
#   24. leak_test.py on a 60k-step Gibbs checkpoint (pred_std/true_std > 0.3)
#   25. m4_pfn_vs_gp_indist.py rho >= 0.5 at 60k
#   26. empirical non-stationarity check
#   27. m5_bo_benchmark.py --target fea -> mean regret >= C2's 0.013
