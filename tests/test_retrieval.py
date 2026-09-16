from app.retrieval import reciprocal_rank_fusion, tokenize


def test_shared_hit_outranks_single_source():
    assert reciprocal_rank_fusion([["dense", "both"], ["keyword", "both"]], 3)[0] == "both"


def test_duplicate_does_not_inflate_and_limit_holds():
    assert reciprocal_rank_fusion([["a", "a", "b"], ["b"]], 1) == ["b"]
    assert reciprocal_rank_fusion([], 4) == []


def test_keyword_numbers_and_case():
    assert tokenize("Probe A-12: 0.5 C") == ["probe", "a", "12", "0", "5", "c"]
