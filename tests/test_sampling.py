
import torch


def fused_sampling(logits, temperature=1.0, top_p=0.9, top_k=-1):
    """Fused sampling logic to be tested."""
    # 1. Greedy decoding (Argmax) optimization
    if temperature < 1e-5 or top_p < 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Fuse Temperature + Softmax
    if temperature != 1.0:
        logits = logits / temperature

    # 3. Top-K Filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")

    # 4. Top-P (Nucleus) Filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    # 5. Final Sampling
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def test_argmax():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    # Low temp -> Argmax
    out = fused_sampling(logits, temperature=0.0)
    assert out.item() == 2

    # Low top_p -> Argmax (effectively)
    out = fused_sampling(logits, top_p=0.0)
    assert out.item() == 2

def test_top_k():
    torch.manual_seed(42)
    logits = torch.tensor([[10.0, 9.0, 1.0, 2.0]])
    # Top-k=1 -> should pick index 0
    out = fused_sampling(logits, top_k=1, temperature=1.0, top_p=1.0)
    assert out.item() == 0

    # Top-k=2 -> should pick 0 or 1
    # with these logits, 0 and 1 are much higher, so likely one of them.
    # Set logits so that 2 and 3 are -inf if filtered
    logits = torch.tensor([[10.0, 10.0, 0.0, 0.0]])
    out = fused_sampling(logits, top_k=2, temperature=1.0, top_p=1.0)
    assert out.item() in [0, 1]

def test_top_p():
    torch.manual_seed(42)
    # 0.6, 0.3, 0.1
    # top_p=0.5 -> Only 0.6 should be selected?
    # Logic: cumsum [0.6, 0.9, 1.0]. >0.5? [T, T, T]. Shift: [F, T, T]. Remove indices 1 and 2.
    # So only index 0 remains.
    
    logits = torch.tensor([[1.0, 0.0, (-10.0)]])
    # softmax([1, 0, -10]) approx [0.73, 0.27, 0.0]
    
    # Let's use exact probs for easier reasoning
    # logits = log(probs)
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    logits = torch.log(probs)
    
    # top_p = 0.6. cumsum=[0.7, 0.9, 1.0]. >0.6? [T, T, T]. Shift [F, T, T]. Keep 0.
    out = fused_sampling(logits, top_p=0.6, temperature=1.0)
    assert out.item() == 0
    
    # top_p = 0.8. cumsum=[0.7, 0.9, 1.0]. >0.8? [F, T, T]. Shift [F, F, T]. Keep 0, 1.
    # Can return 0 or 1.
    results = []
    for _ in range(20):
        results.append(fused_sampling(logits, top_p=0.8, temperature=1.0).item())
    assert 0 in results or 1 in results
    assert 2 not in results

if __name__ == "__main__":
    test_argmax()
    test_top_k()
    test_top_p()
    print("All tests passed")
