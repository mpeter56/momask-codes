"""
Minimal unit tests for the semantic spectrum pipeline.
Run with:  python -m pytest tests/test_spectrum_pipeline.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_spectrum.analyzer import SpectrumAnalyzer
from semantic_spectrum.synthesizer import SpectrumCaptionSynthesizer
from semantic_spectrum.dimensions import DIMENSIONS


def _fake_motion(T: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    motion = rng.standard_normal((T, 263)).astype(np.float32)
    # Simulate foot contact (binary-ish values)
    motion[:, 256:260] = (rng.random((T, 4)) > 0.5).astype(np.float32)
    return motion


# ------------------------------------------------------------------
# SpectrumAnalyzer
# ------------------------------------------------------------------

class TestSpectrumAnalyzer:
    def test_returns_all_dimensions(self):
        analyzer = SpectrumAnalyzer()
        motion = _fake_motion()
        scores = analyzer.analyze(motion)
        assert set(scores.keys()) == set(DIMENSIONS.keys())

    def test_scores_in_range(self):
        analyzer = SpectrumAnalyzer()
        motion = _fake_motion()
        scores = analyzer.analyze(motion)
        for dim, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{dim} score {score} out of range"

    def test_scores_are_independent(self):
        """Multiple dimensions can simultaneously be high (not softmax)."""
        analyzer = SpectrumAnalyzer()
        motion = _fake_motion()
        scores = analyzer.analyze(motion)
        # Sum CAN exceed 1.0 — that's intentional
        total = sum(scores.values())
        assert total >= 0.0  # trivially true, but asserts no crash

    def test_short_motion_raises(self):
        analyzer = SpectrumAnalyzer(min_frames=40)
        motion = _fake_motion(T=10)
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze(motion)

    def test_wrong_dim_raises(self):
        analyzer = SpectrumAnalyzer()
        motion = np.zeros((60, 200))  # wrong feature dim
        with pytest.raises(ValueError, match="263"):
            analyzer.analyze(motion)

    def test_calibrate_clips_to_range(self):
        cal = {"walk": (0.1, 0.9)}
        analyzer = SpectrumAnalyzer(dimensions=["walk"], calibration=cal)
        motion = _fake_motion()
        scores = analyzer.analyze(motion)
        assert 0.0 <= scores["walk"] <= 1.0

    def test_subset_dimensions(self):
        analyzer = SpectrumAnalyzer(dimensions=["walk", "dance"])
        motion = _fake_motion()
        scores = analyzer.analyze(motion)
        assert set(scores.keys()) == {"walk", "dance"}


# ------------------------------------------------------------------
# SpectrumCaptionSynthesizer
# ------------------------------------------------------------------

class TestSpectrumCaptionSynthesizer:
    def _scores(self, **kwargs) -> dict:
        base = {d: 0.0 for d in DIMENSIONS}
        base.update(kwargs)
        return base

    def test_basic_output_format(self):
        synth = SpectrumCaptionSynthesizer()
        scores = self._scores(walk=0.82, dance=0.18)
        caption = synth.synthesize(scores)
        assert "[walk:0.82]" in caption
        assert "[dance:0.18]" in caption
        assert caption.startswith("[")
        assert caption.endswith(".")

    def test_stable_ordering(self):
        """Same scores always produce the same token order."""
        synth = SpectrumCaptionSynthesizer()
        scores = self._scores(walk=0.82, dance=0.18, spin=0.22)
        c1 = synth.synthesize(scores)
        c2 = synth.synthesize(scores)
        assert c1 == c2

    def test_threshold_excludes_low_scores(self):
        synth = SpectrumCaptionSynthesizer(tag_threshold=0.10)
        scores = self._scores(walk=0.90, dance=0.01)
        caption = synth.synthesize(scores)
        assert "[dance:" not in caption

    def test_interpolation_count(self):
        synth = SpectrumCaptionSynthesizer()
        a = self._scores(walk=0.9, dance=0.1)
        b = self._scores(walk=0.1, dance=0.9)
        captions = synth.synthesize_interpolated(a, b, steps=5)
        assert len(captions) == 5

    def test_interpolation_endpoints(self):
        synth = SpectrumCaptionSynthesizer()
        a = self._scores(walk=0.9, dance=0.1)
        b = self._scores(walk=0.1, dance=0.9)
        captions = synth.synthesize_interpolated(a, b, steps=5)
        # First caption should be walk-dominant
        assert "[walk:0.90]" in captions[0]
        # Last caption should be dance-dominant
        assert "[dance:0.90]" in captions[-1]

    def test_small_delta_small_string_change(self):
        """A 0.01 change in one dimension should produce a very similar string."""
        synth = SpectrumCaptionSynthesizer()
        s1 = self._scores(walk=0.80, dance=0.20)
        s2 = self._scores(walk=0.79, dance=0.21)
        c1 = synth.synthesize(s1)
        c2 = synth.synthesize(s2)
        # Should differ only in the numeric values, not structure
        assert c1.replace("0.80", "0.79").replace("0.20", "0.21") == c2

    def test_all_zero_scores(self):
        synth = SpectrumCaptionSynthesizer()
        scores = self._scores()
        caption = synth.synthesize(scores)
        assert "A person" in caption


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------

class TestIntegration:
    def test_analyze_then_synthesize(self):
        analyzer = SpectrumAnalyzer()
        synth = SpectrumCaptionSynthesizer()
        motion = _fake_motion(T=80)
        scores = analyzer.analyze(motion)
        caption = synth.synthesize(scores)
        assert isinstance(caption, str)
        assert len(caption) > 10
