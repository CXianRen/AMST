# This file is for testing the fusion modules we used 
# in the project.

from .fusion_modules import MLASum, \
    EarlySum, EarlyConcat, LateSum, \
    newGatedFusion, newFiLM, gen_fusion_v2

import torch 
    
# test gen_fusion_v2
def test_gen_fusion_v2():
    args = type("args", (object,), {})()
    args.fusion_method = "esum"
    
    name_list = ["a", "v"]
    
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, EarlySum), "Fusion method should be EarlySum"
    
    
    args.fusion_method = "concat"
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, EarlyConcat), "Fusion method should be EarlyConcat"
    
    args.fusion_method = "lsum"
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, LateSum), "Fusion method should be LateSum"
    
    args.fusion_method = "msum"
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, MLASum), "Fusion method should be MLASum"
    
    args.fusion_method = "gated"
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, newGatedFusion), "Fusion method should be newGatedFusion"
    
    args.fusion_method = "gate"
    # for backward compatibility
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, newGatedFusion), "Fusion method should be newGatedFusion"
    
    args.fusion_method = "film"
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, newFiLM), "Fusion method should be newFiLM"
    
    print("passed test_gen_fusion_v2")
    
# test early sum
def test_fusion_method_2_modalities(fusion_method, expected_class):
    args = type("args", (object,), {})()
    args.fusion_method = fusion_method
    
    name_list = ["a", "v"]
    
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, expected_class), f"Fusion method should be {expected_class.__name__}"
    
    assert fusion_obj.n_modalities == 2, "Number of modalities should be 2"
    
    # test forward, batchsize 1
    a_embed = torch.zeros(1, 512)
    v_embed = torch.zeros(1, 512)
    
    embeddings_dict = {"a": a_embed, "v": v_embed}
    output = fusion_obj(embeddings_dict)
    
    assert output.shape == (1, 6), \
        "Output shape should be (1, 6)"
    assert len(fusion_obj.out_dict) == 2, \
        "Output dict should have 2 keys"
    assert "a" in fusion_obj.out_dict, \
        "Output dict should have key 'a'"
    assert "v" in fusion_obj.out_dict, \
        "Output dict should have key 'v'"
    assert fusion_obj.out_dict["a"].shape == (1, 6), \
        "Output dict 'a' should have shape (1, 6)"
    assert fusion_obj.out_dict["v"].shape == (1, 6), \
        "Output dict 'v' should have shape (1, 6)"
    
    # test get_out_m
    out_a = fusion_obj.get_out_m('a')
    assert out_a.shape == (1, 6), \
        "out_m shape should be (1, 6)"   
    assert torch.equal(out_a, fusion_obj.out_dict["a"]), \
        "out_m should be equal to out_dict['a']"
    
    out_v = fusion_obj.get_out_m('v')
    assert out_v.shape == (1, 6), \
        "out_m shape should be (1, 6)"
    assert torch.equal(out_v, fusion_obj.out_dict["v"]), \
        "out_m should be equal to out_dict['v']"

    # expecting error
    try:
        out_m = fusion_obj.get_out_m('x')
        assert False, "get_out_m should raise KeyError"
    except ValueError:
        pass
    
def test_fusion_method_3_modalities(fusion_method, expected_class):
    args = type("args", (object,), {})()
    args.fusion_method = fusion_method
    
    name_list = ["a", "v", "t"]
    
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, expected_class), \
        f"Fusion method should be {expected_class.__name__}"
    
    assert fusion_obj.n_modalities == 3, \
        "Number of modalities should be 3"
    
    # test forward, batchsize 1
    a_embed = torch.zeros(1, 512)
    v_embed = torch.zeros(1, 512)
    t_embed = torch.zeros(1, 512)
    
    embeddings_dict = {"a": a_embed, "v": v_embed, "t": t_embed}
    output = fusion_obj(embeddings_dict)
    
    assert output.shape == (1, 6), \
        "Output shape should be (1, 6)"
        
def test_early_sum():
    test_fusion_method_2_modalities("esum", EarlySum)
    test_fusion_method_3_modalities("esum", EarlySum)
    print("passed test_early_sum")

def test_late_sum():
    test_fusion_method_2_modalities("lsum", LateSum)
    test_fusion_method_3_modalities("lsum", LateSum)
    print("passed test_late_sum")

def test_early_concat():
    test_fusion_method_2_modalities("concat", EarlyConcat)
    test_fusion_method_3_modalities("concat", EarlyConcat)
    print("passed test_early_concat")
    
def test_mlasum():
    test_fusion_method_2_modalities("msum", MLASum)
    test_fusion_method_3_modalities("msum", MLASum)
    print("passed test_mlasum")


def test_gated_film_fusion(fusion_method, expected_class):
    args = type("args", (object,), {})()
    args.fusion_method = fusion_method
    
    name_list = ["a", "v"]
    
    fusion_obj = gen_fusion_v2(args, 
                    input_dim=512, 
                    output_dim=6,
                    name_list=name_list)
    assert isinstance(fusion_obj, expected_class), f"Fusion method should be {expected_class.__name__}"
        
    # test forward, batchsize 1
    a_embed = torch.zeros(1, 512)
    v_embed = torch.zeros(1, 512)
    
    embeddings_dict = {"a": a_embed, "v": v_embed}
    output = fusion_obj(embeddings_dict)
    
    assert output.shape == (1, 6), \
        "Output shape should be (1, 6)"
    
    # test get_out_m
    try:
        out_m = fusion_obj.get_out_m('a')
        assert False, "get_out_m should raise KeyError"
    except NotImplementedError:
        pass
    
    try:
        out_m = fusion_obj.get_out_m('v')
        assert False, "get_out_m should raise KeyError"
    except NotImplementedError:
        pass

def test_newGatedFusion():
    test_gated_film_fusion("gated", newGatedFusion)
    print("passed test_newGatedFusion")

def test_newFiLM():
    test_gated_film_fusion("film", newFiLM)
    print("passed test_newFiLM")


if __name__ == "__main__":
    # Run all tests
    test_gen_fusion_v2()
    test_early_sum()
    test_late_sum()
    test_early_concat()
    test_mlasum()
    test_newGatedFusion()
    test_newFiLM()
    # test all fusion methods
