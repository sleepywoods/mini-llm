"""졸업 프로젝트 환경(transformers 4.45.1, CPU torch)에서 Hub 모델 검증
  0: 로드 + 토크나이저 + greedy 생성
  A: HF fx symbolic_trace (prefill)
  B: torch.export, 실제 텐서, prefill, use_cache=True (+ 결과 일치 확인)
  C: torch.export, FakeTensor, prefill, use_cache=True
  D: torch.export, FakeTensor, decode (토큰 1개 + KV 캐시), use_cache=True
paths를 import하지 않음 → 졸업 프로젝트와 같은 기본 캐시(C) 사용
"""
import collections
import sys
import traceback

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO = "skmbkang/tinystories-gpt2-30m"
T_PREFILL = 32          # prefill 입력 길이
V = 50257


def section(title):
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def run(title, fn):
    """각 테스트를 독립 실행: 하나가 실패해도 나머지는 계속"""
    section(title)
    try:
        fn()
        print("→ 성공")
    except Exception as e:
        print(f"→ 실패: {type(e).__name__}: {str(e)[:400]}")
        traceback.print_exc(limit=3)


def shape_of(node):
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    if isinstance(val, (tuple, list)) and val and isinstance(val[0], torch.Tensor):
        return tuple(val[0].shape)
    return None


def summarize_export(ep, top=12):
    nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
    cnt = collections.Counter(str(n.target) for n in nodes)
    print(f"연산 노드 수: {len(nodes)}")
    for k, v in cnt.most_common(top):
        print(f"   {k:55s} {v}")
    # 어텐션 연산의 입력 shape (prefill vs decode 차이 확인용)
    for n in nodes:
        name = str(n.target)
        if "scaled_dot_product" in name or name.startswith("aten.bmm") or name.startswith("aten.matmul"):
            in_shapes = [shape_of(a) for a in n.args if isinstance(a, torch.fx.Node)]
            print(f"첫 어텐션 연산: {name}  입력 shape {in_shapes}")
            break


def cache_info(pkv):
    if isinstance(pkv, tuple):
        return f"tuple (레이어 {len(pkv)}개, K shape {tuple(pkv[0][0].shape)})"
    return type(pkv).__name__


# ---------------------------------------------------------------- 0
print("python      :", sys.executable)
print("torch       :", torch.__version__, "/ transformers", transformers.__version__)

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO).eval()
config = AutoConfig.from_pretrained(REPO)


def test_load():
    n = sum(p.numel() for p in model.parameters())
    print(f"로드: {type(tok).__name__}, {type(model).__name__}, 파라미터 {n:,}")
    print("tied:", model.lm_head.weight is model.transformer.wte.weight)
    print("attention 구현:", model.config._attn_implementation)
    ids = tok("Once upon a time", return_tensors="pt")
    print("토큰 ID:", ids["input_ids"].tolist(), "(기대값 [[7454, 2402, 257, 640]])")
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=50256)
    print("greedy:", tok.decode(g[0], skip_special_tokens=True))


run("0. 로드 + 생성", test_load)


# ---------------------------------------------------------------- A
def test_fx():
    from transformers.utils.fx import symbolic_trace
    gm = symbolic_trace(model, input_names=["input_ids"])
    cnt = collections.Counter()
    for node in gm.graph.nodes:
        if node.op == "call_module":
            cnt[f"module   {type(gm.get_submodule(node.target)).__name__}"] += 1
        elif node.op in ("call_function", "call_method"):
            cnt[f"{node.op[5:]:8s} {getattr(node.target, '__name__', node.target)}"] += 1
    print(f"fx 노드 수: {len(gm.graph.nodes)}")
    for k, v in cnt.most_common(12):
        print(f"   {k:40s} {v}")
    x = torch.randint(0, V, (1, T_PREFILL))
    with torch.no_grad():
        a = model(input_ids=x).logits
        b = gm(input_ids=x)["logits"]
    print("추적 결과 일치:", torch.allclose(a, b, atol=1e-5), f"(최대 차이 {(a - b).abs().max().item():.2e})")


run("A. HF fx symbolic_trace (prefill)", test_fx)


# ---------------------------------------------------------------- B
def test_export_real():
    x = torch.randint(0, V, (1, T_PREFILL))                       # (1, T)
    kw = {"input_ids": x, "use_cache": True}
    with torch.no_grad():
        ref = model(**kw)
    print("캐시 형식:", cache_info(ref.past_key_values))
    ep = torch.export.export(model, args=(), kwargs=kw, strict=False)
    summarize_export(ep)
    with torch.no_grad():
        out = ep.module()(**kw)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    print("export 결과 일치:", torch.allclose(ref.logits, logits, atol=1e-5))


run("B. torch.export 실제 텐서 / prefill / use_cache=True", test_export_real)


# ---------------------------------------------------------------- C, D (FakeTensor)
from torch._subclasses.fake_tensor import FakeTensorMode

fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
state = {}


def test_export_fake_prefill():
    with fake_mode:
        fmodel = AutoModelForCausalLM.from_config(config).eval()  # 파라미터가 FakeTensor (실제 숫자 없음)
        print("파라미터 타입:", type(next(fmodel.parameters())).__name__)
        x = torch.randint(0, V, (1, T_PREFILL))                    # FakeTensor (1, T)
        kw = {"input_ids": x, "use_cache": True}
        ep = torch.export.export(fmodel, args=(), kwargs=kw, strict=False)
        out = fmodel(**kw)                                          # D에서 쓸 KV 캐시를 FakeTensor로 생성
    state["fmodel"], state["pkv"] = fmodel, out.past_key_values
    print("캐시 형식:", cache_info(out.past_key_values))
    summarize_export(ep)


def test_export_fake_decode():
    assert "pkv" in state, "C가 실패해서 KV 캐시가 없음"
    with fake_mode:
        x1 = torch.randint(0, V, (1, 1))                            # 새 토큰 1개 (1, 1)
        kw = {"input_ids": x1, "past_key_values": state["pkv"], "use_cache": True}
        ep = torch.export.export(state["fmodel"], args=(), kwargs=kw, strict=False)
    summarize_export(ep)


run("C. torch.export FakeTensor / prefill / use_cache=True", test_export_fake_prefill)
run("D. torch.export FakeTensor / decode (1토큰 + KV 캐시)", test_export_fake_decode)