import math
import numpy as np
import pytest
from ivon_lora.metrics import classification_metrics, reliability_bins, average_probabilities
from ivon_lora.report import paired_deltas


def test_known_calibration_example():
    # Both confidences are .8, exactly one is correct: ECE = .3.
    result = classification_metrics([[.8,.2],[.2,.8]], np.array([0,0]))
    assert result["accuracy"] == .5
    assert result["ece"] == pytest.approx(.3)
    assert result["nll"] == pytest.approx(-.5*(math.log(.8)+math.log(.2)))
    assert result["brier"] == pytest.approx(.68)


def test_confidence_one_is_included():
    p = np.eye(3)
    result = classification_metrics(p, np.array([0,1,2]))
    assert result["accuracy"] == 1
    assert result["ece"] == 0
    assert result["nll"] == 0
    assert sum(b["count"] for b in reliability_bins(p,np.array([0,1,2]))) == 3


@pytest.mark.parametrize("p,y", [([[.8,.8]],[0]), ([[float('nan'),0]],[0]), ([[.2,.8]],[2]), ([],[])])
def test_invalid_probabilities_rejected(p,y):
    with pytest.raises(ValueError):
        classification_metrics(p,np.array(y,dtype=int))


def test_probability_ensemble_and_paired_seeds():
    np.testing.assert_allclose(average_probabilities([[[.9,.1]],[[.3,.7]]]), [[.6,.4]])
    records = [{"seed":21,"method":"adamw","ece":.3}, {"seed":42,"method":"adamw","ece":.4},
               {"seed":21,"method":"ivon_mc","ece":.2},{"seed":42,"method":"ivon_mc","ece":.2}]
    assert paired_deltas(records,"ece","ivon_mc")["mean"] == pytest.approx(-.15)
    with pytest.raises(ValueError):
        paired_deltas(records[:-1],"ece","ivon_mc")
