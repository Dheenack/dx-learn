from dxlearn.core.scoring import multi_objective_score

def test_score():
    s = multi_objective_score(0.9, 0.8, 10)
    assert s > 0
