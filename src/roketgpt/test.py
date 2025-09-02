# === test.py — Gerçekçi değerlendirme (generatif, teacher forcing YOK) ===
# - Checkpoint'i yükler
# - input.txt + veri_properties.txt'den QA çiftleri üretir
# - Train/Val ayrımı yapar (ayrık), val setinde üretim yapıp EM ve karakter isabeti hesaplar

import os, re, random, unicodedata
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from modelroketsan import GPTLanguageModel


# ----------------- Yardımcılar -----------------
_TMAP = str.maketrans({'ı':'i','İ':'i','I':'i','ç':'c','Ç':'c','ğ':'g','Ğ':'g','ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'})
def norm(s: str) -> str:
    s = s.strip().casefold().translate(_TMAP)
    s = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))

def canon_name(name: str) -> str:
    return re.sub(r'\s*\(.+?\)\s*', '', name).strip()


# ----------------- Veri yolları -----------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = str(BASE_DIR / 'input.txt')
PROPS_PATH = str(BASE_DIR / 'veri_properties.txt')

def resolve_ckpt() -> str:
    env_p = os.environ.get('DOMAIN_LLM_CKPT', '').strip() or os.environ.get('GENERAL_LLM_CKPT', '').strip()
    if env_p and os.path.exists(env_p):
        return env_p
    for p in [BASE_DIR / 'gpt_gen_checkpoint.pth', Path.cwd() / 'gpt_gen_checkpoint.pth', Path('/content/gpt_gen_checkpoint.pth')]:
        if p.exists():
            return str(p)
    # Son çare: kullanıcı konumu
    fallback = Path('/content/Untitled/gpt_gen_checkpoint.pth')
    return str(fallback) if fallback.exists() else ''


# ----------------- input.txt -> tanımlar -----------------
raw = Path(INPUT_PATH).read_text(encoding='utf-8').strip()
defs = {}
for blk in re.split(r'\n{2,}', raw):
    if ':' in blk:
        term, defn = blk.split(':', 1)
        defs[term.strip()] = ' '.join(defn.strip().splitlines())


# ----------------- veri_properties.txt -> kayıtlar -----------------
props_text = Path(PROPS_PATH).read_text(encoding='utf-8').strip()
prop_db = {}
current = None
for line in props_text.splitlines():
    if not line.strip():
        continue
    m = re.match(r'^\s*([^:]+?):\s*$', line)
    if m and '(' in m.group(1):
        current = m.group(1).strip()
        if current not in prop_db:
            prop_db[current] = {}
        continue
    m2 = re.match(r'^\s*([^:]+?)\s*:\s*(.+)$', line)
    if m2 and current:
        key, value = m2.group(1).strip(), m2.group(2).strip()
        if key.lower().startswith('özellik'):
            items = [i.strip(' .;') for i in re.split(r',|;|\n|\||/|•', value) if i.strip()]
            prop_db[current].setdefault('Özellikler', []).extend(items)
        else:
            prop_db[current][key] = value


def serialize_record(name, fields):
    parts = [f"{canon_name(name)}:"]
    for k, v in fields.items():
        if isinstance(v, list): parts.append(f"{k}: {', '.join(v)}")
        else: parts.append(f"{k}: {v}")
    return ' '.join(parts)


# ----------------- QA çiftleri -----------------
SEP = '</s>'
def build_qa_pairs() -> List[Tuple[str, str]]:
    data = []
    # Tanım soruları
    def_qv = ["nedir?", "ne demektir?", "hakkında bilgi ver."]
    for term, definition in defs.items():
        q = f"{term} {random.choice(def_qv)}"
        a = definition
        data.append((q, a))
    # Özellik soruları
    feat_qv = ["özellikleri nelerdir?", "hangi özelliklere sahip?", "özeti nedir?"]
    for full, fields in prop_db.items():
        base = canon_name(full)
        a_parts = []
        for k, v in fields.items():
            if isinstance(v, list): a_parts.append(f"{k}: {', '.join(v)}")
            else: a_parts.append(f"{k}: {v}")
        a = f"{base} — " + '; '.join(a_parts) if a_parts else base
        q = f"{base} {random.choice(feat_qv)}"
        data.append((q, a))
    random.shuffle(data)
    return data


def top_k_records(query: str, k=6) -> List[str]:
    nq = norm(query)
    hits = []
    # 1) Direkt terim eşleşmeleri öncelikli
    for name in list(defs.keys()) + list(prop_db.keys()):
        if norm(canon_name(name)) in nq or norm(name) in nq:
            hits.append(canon_name(name))
    # 2) kaba anahtar kelime
    toks = [t for t in re.split(r'[\s,;:]+', nq) if t]
    for name in prop_db.keys():
        if any(t in norm(name) for t in toks):
            hits.append(canon_name(name))
    uniq = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq[:k]


@torch.no_grad()
def generate_answer(model, stoi, itos, block_size, device, question: str) -> str:
    # Bağlamı kur
    cands = top_k_records(question, k=6)
    ctx_lines = []
    for full, fields in prop_db.items():
        base = canon_name(full)
        if base in cands:
            ctx_lines.append(serialize_record(full, fields))
    ctx = ' | '.join(ctx_lines)

    prompt = f"Soru: {question} {SEP} Bağlam: {ctx} {SEP} Cevap: "
    def encode(s):
        any_idx = next(iter(stoi.values()))
        return [stoi.get(c, any_idx) for c in s]
    def decode(l):
        return ''.join(itos.get(i, '') for i in l)

    start = '<s>' + prompt
    idx = torch.tensor([encode(start)], dtype=torch.long).to(device)
    max_new_tokens = 196
    top_k = 50
    temperature = 0.9

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits / max(1e-5, temperature), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
        # Erken durdurma: ayıraç görürsek kesmeye izin ver
        if itos.get(int(next_token.item()), '') == '\n':
            pass

    txt = decode(idx[0].tolist())
    gen = txt[len('<s>'):]
    # Cevap bölümünü çıkar
    lower = gen.lower()
    pos = lower.rfind('cevap:')
    ans = gen[pos+len('cevap:'):] if pos != -1 else gen
    ans = ans.split(SEP)[0] if SEP in ans else ans
    return ans.strip()


def exact_match(pred: str, gold: str) -> bool:
    return norm(pred) == norm(gold)

def char_accuracy(pred: str, gold: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    # karakter-düzeyinde hizalama (sade): kısa uzunluk üzerinden
    n = max(len(gold), 1)
    m = min(len(gold), len(pred))
    correct = sum(1 for i in range(m) if pred[i] == gold[i])
    return correct / n


def main():
    ckpt_path = resolve_ckpt()
    if not ckpt_path:
        print('❌ Checkpoint bulunamadı.')
        return
    print(f'ℹ️ Checkpoint otomatik bulundu: {ckpt_path}')

    # Checkpoint yükle
    ckpt = torch.load(ckpt_path, map_location='cpu')
    itos = ckpt['itos']; stoi = ckpt['stoi']
    cfg = ckpt.get('config', {})
    block_size = cfg.get('block_size', 256)
    n_embd  = cfg.get('n_embd', 512)
    n_layer = cfg.get('n_layer', 8)
    n_head  = cfg.get('n_head', 8)
    dropout = cfg.get('dropout', 0.1)
    print(f'✅ Checkpoint: {ckpt_path}')
    print(f'🔧 Mimari ⇒ block_size={block_size}, n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}')

    model = GPTLanguageModel(
        vocab_size=len(itos),
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout
    ).to('cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # QA verisini kur ve böl
    pairs = build_qa_pairs()
    print(f'📚 QA örnekleri: {len(pairs)}')
    if len(pairs) < 4:
        print('⚠️ Yetersiz örnek; değerlendirme atlanyor.')
        return

    random.seed(42)
    random.shuffle(pairs)
    val_ratio = 0.1
    cut = max(1, int(len(pairs) * (1.0 - val_ratio)))
    train_pairs = pairs[:cut]
    val_pairs = pairs[cut:]
    print(f'▶ Train: {len(train_pairs)}, ✋ Val: {len(val_pairs)}')

    # Val üzerinde gerçek üretimle ölç
    em_hits = 0
    char_hits = 0.0
    labeled_chars = 0

    for q, gold in val_pairs:
        pred = generate_answer(model, stoi, itos, block_size, 'cpu', q)
        if exact_match(pred, gold):
            em_hits += 1
        # char-acc (sade)
        acc = char_accuracy(pred, gold)
        char_hits += acc * max(len(gold), 1)
        labeled_chars += max(len(gold), 1)

    em = (em_hits / max(1, len(val_pairs))) * 100.0
    tok_acc = (char_hits / max(1, labeled_chars)) * 100.0
    print(f"[Eval] EM={em:.2f}%  Char-Acc={tok_acc:.2f}%  (N={len(val_pairs)}, labeled={labeled_chars})")


if __name__ == '__main__':
    main()

# === test_model.py (v1.2: checkpoint otomatik bul + mimariyi checkpoint'ten oku + ölçümler) ===
import argparse, time, re, unicodedata, os, glob
from pathlib import Path
import torch
import torch.nn.functional as F

from modelroketsan import GPTLanguageModel

# ---------- Argümanlar ----------
def get_args():
    p = argparse.ArgumentParser(description="LLM test & ölçüm betiği")
    p.add_argument("--input_path", type=str, default="/content/Untitled/input.txt")
    p.add_argument("--props_path", type=str, default="/content/Untitled/veri_properties.txt")
    p.add_argument("--ckpt_path",  type=str, default="/content/gpt_qa_checkpoint.pth",
                   help="Checkpoint .pth dosyası (bulunamazsa otomatik taranır)")
    p.add_argument("--block_size", type=int, default=128, help="(Fallback) Checkpoint'te yoksa kullanılır")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_passes", type=int, default=1)
    p.add_argument("--gen_tokens", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args()

# ---------- Yardımcılar (train.py ile uyumlu) ----------
_TMAP = str.maketrans({'ı':'i','İ':'i','I':'i','ç':'c','Ç':'c','ğ':'g','Ğ':'g','ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'})
def norm(s: str) -> str:
    import unicodedata
    s = s.strip().casefold().translate(_TMAP)
    s = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))

def canon_name(name: str) -> str:
    import re
    return re.sub(r'\s*\(.+?\)\s*', '', name).strip()

def parse_input_defs(input_path: str):
    raw = Path(input_path).read_text(encoding='utf-8').strip()
    defs = {}
    for blk in raw.split('\n\n'):
        if ':' in blk:
            term, defn = blk.split(':', 1)
            defs[term.strip()] = ' '.join(defn.strip().splitlines())
    return defs

def parse_props(props_path: str):
    import re
    text = Path(props_path).read_text(encoding='utf-8').strip()
    prop_db = {}
    current = None
    for line in text.splitlines():
        if not line.strip(): 
            continue
        m = re.match(r'^\s*([^:]+?):\s*$', line)
        if m and '(' in m.group(1):
            current = m.group(1).strip()
            if current not in prop_db: prop_db[current] = {}
            continue
        m2 = re.match(r'^\s*([^:]+?)\s*:\s*(.+)$', line)
        if m2 and current:
            key, value = m2.group(1).strip(), m2.group(2).strip()
            if key.lower().startswith('özellik'):
                items = [i.strip(' .;') for i in re.split(r',|;|\n', value) if i.strip()]
                prop_db[current].setdefault('Özellikler', []).extend(items)
            else:
                prop_db[current][key] = value
    return prop_db

def build_feature_index(prop_db: dict):
    feature2terms = {}
    for full, fields in prop_db.items():
        base = canon_name(full)
        feats = []
        for k, v in fields.items():
            if isinstance(v, list): feats.extend(v)
            else: feats.append(v)
        for f in feats:
            nf = norm(f)
            if nf:
                feature2terms.setdefault(nf, set()).add(base)
    return feature2terms

def build_qa_examples(defs: dict, prop_db: dict, feature2terms: dict):
    qa_examples = []
    # Tanım QA
    for term, definition in defs.items():
        qa_examples.append({
            'question': f"{term} nedir?",
            'context': definition,
            'answer_start': 0,
            'answer_text': definition
        })
    # Özellik QA
    for full, fields in prop_db.items():
        base = canon_name(full)
        if not fields: 
            continue
        parts = []
        for k, v in fields.items():
            if isinstance(v, list): parts.append(f"{k}: {', '.join(v)}")
            else: parts.append(f"{k}: {v}")
        answer = " | ".join(parts)
        for q in [f"{base} özellikleri nelerdir?",
                  f"{base}’ın özellikleri nelerdir?",
                  f"{base} ozellikleri neler?",
                  f"{base} hangi özelliklere sahip?"]:
            qa_examples.append({
                'question': q,
                'context': answer,
                'answer_start': 0,
                'answer_text': answer
            })
    # Özellikten listeleme
    for nf, terms in feature2terms.items():
        if not terms: 
            continue
        any_original = None
        for full, fields in prop_db.items():
            feats = []
            for k,v in fields.items():
                feats.extend(v if isinstance(v,list) else [v])
            for f in feats:
                if norm(f) == nf:
                    any_original = f; break
            if any_original: break
        feature_str = any_original or nf
        answer = ", ".join(sorted(terms))
        for q in [f"Hangi sistemlerde {feature_str} var?",
                  f"{feature_str} özelliğine sahip olanlar hangileri?",
                  f"{feature_str} kimlerde bulunur?"]:
            qa_examples.append({
                'question': q,
                'context': answer,
                'answer_start': 0,
                'answer_text': answer
            })
    return qa_examples

def make_dataset(qa_examples, stoi, block_size):
    sep_id = stoi.get('</s>', None)
    pad_id = stoi.get('<pad>', 0)
    bos_id = stoi.get('<s>', None)

    def safe_encode(s: str):
        return [stoi.get(c, pad_id) for c in s]

    data = []
    for ex in qa_examples:
        q_ids = safe_encode(ex['question'])
        c_ids = safe_encode(ex['context'])
        a_ids = safe_encode(ex['answer_text'])
        inp = ([] if bos_id is None else [bos_id]) + q_ids + ([] if sep_id is None else [sep_id]) + c_ids
        if len(inp) > block_size:
            inp = inp[-block_size:]
        lbl = [-100] * len(inp)
        ans_tok_start = (0 if bos_id is None else 1) + len(q_ids) + (0 if sep_id is None else 1) + ex['answer_start']
        for i, tid in enumerate(a_ids):
            pos = ans_tok_start + i
            if 0 <= pos < len(lbl):
                lbl[pos] = tid
        if len(inp) < block_size:
            pad_len = block_size - len(inp)
            inp = [pad_id]*pad_len + inp
            lbl = [-100]*pad_len + lbl
        label_positions = [i for i,t in enumerate(lbl) if t != -100]
        data.append((
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(lbl, dtype=torch.long),
            label_positions
        ))
    return data

def split_dataset(processed, train_ratio=0.9):
    n = len(processed)
    train_size = int(train_ratio * n)
    return processed[:train_size], processed[train_size:]

# ---------- Ölçüm ----------
@torch.no_grad()
def evaluate_full(model, dataset, batch_size=64, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_label = 0
    total_correct = 0
    total_em = 0
    N = len(dataset)

    for i in range(0, N, batch_size):
        batch = dataset[i:i+batch_size]
        x = torch.stack([b[0] for b in batch]).to(device)
        y = torch.stack([b[1] for b in batch]).to(device)
        logits, loss = model(x, y)
        total_loss += loss.item() * x.size(0)

        pred = logits.argmax(dim=-1)
        mask = (y != -100)
        total_label += mask.sum().item()
        total_correct += (pred[mask] == y[mask]).sum().item()

        for bi in range(x.size(0)):
            pos = [idx for idx in batch[bi][2] if idx < logits.size(1)]
            if not pos:
                continue
            gt_seq = y[bi, pos].tolist()
            pr_seq = pred[bi, pos].tolist()
            total_em += int(pr_seq == gt_seq)

    avg_loss = total_loss / max(1, N)
    token_acc = (total_correct / max(1, total_label)) * 100.0
    em = (total_em / max(1, N)) * 100.0
    return {
        "val_loss": avg_loss,
        "token_acc": token_acc,
        "exact_match": em,
        "num_samples": N,
        "num_labeled_tokens": total_label
    }

@torch.no_grad()
def cpu_latency_test(model, sample_batch, gen_prompt_ids, gen_tokens=64):
    device_cpu = torch.device('cpu')
    model_cpu = model.to(device_cpu).eval()

    xb = sample_batch.clone().to(device_cpu)
    runs = 30
    for _ in range(5):
        model_cpu(xb, None)

    t0 = time.perf_counter()
    for _ in range(runs):
        model_cpu(xb, None)
    t1 = time.perf_counter()
    avg_forward_ms = ((t1 - t0) / runs) * 1000.0
    tokens_per_forward = xb.numel()
    throughput_tok_per_s = (tokens_per_forward / (avg_forward_ms / 1000.0))

    gen_inp = gen_prompt_ids.unsqueeze(0).to(device_cpu)
    if gen_inp.size(1) > model_cpu.block_size:
        gen_inp = gen_inp[:, -model_cpu.block_size:]
    elif gen_inp.size(1) < model_cpu.block_size:
        pad_len = model_cpu.block_size - gen_inp.size(1)
        gen_inp = torch.cat([torch.full((1,pad_len), 0, dtype=torch.long), gen_inp], dim=1)

    model_cpu.generate(gen_inp.clone(), 4)

    t0 = time.perf_counter()
    model_cpu.generate(gen_inp.clone(), gen_tokens)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    ms_per_token = total_ms / max(1, gen_tokens)

    return {
        "cpu_forward_ms_per_run": avg_forward_ms,
        "cpu_forward_throughput_tokens_per_s": throughput_tok_per_s,
        "cpu_generate_ms_per_token": ms_per_token
    }

# ---------- Checkpoint / Mimari ----------
def infer_arch_from_state_dict(sd, fallback_block_size=128):
    # block_size
    bs = None
    if 'position_embedding_table.weight' in sd:
        bs = sd['position_embedding_table.weight'].shape[0]
    else:
        tril_keys = [k for k in sd.keys() if k.endswith('.tril')]
        if tril_keys:
            bs = sd[tril_keys[0]].shape[0]
    if bs is None:
        bs = fallback_block_size

    # n_embd
    if 'lm_head.weight' in sd:
        n_embd = sd['lm_head.weight'].shape[1]
    elif 'token_embedding_table.weight' in sd:
        n_embd = sd['token_embedding_table.weight'].shape[1]
    else:
        n_embd = 512

    # n_layer
    layer_idxs = []
    for k in sd.keys():
        m = re.search(r'^blocks\.(\d+)\.', k)
        if m:
            layer_idxs.append(int(m.group(1)))
    n_layer = (max(layer_idxs) + 1) if layer_idxs else 8

    # n_head
    n_head = 8
    key0 = 'blocks.0.sa.heads.0.key.weight'
    if key0 in sd:
        head_size = sd[key0].shape[0]
        if head_size > 0:
            n_head = max(1, n_embd // head_size)

    return bs, n_embd, n_layer, n_head

def find_checkpoint(preferred: str):
    """Önce verilen yolu dener; yoksa olası klasörlerde en yeni GEÇERLİ .pth dosyasını bulur."""
    tried = []
    def is_valid(path):
        try:
            obj = torch.load(path, map_location="cpu")
            return isinstance(obj, dict) and all(k in obj for k in ('model_state_dict','itos','stoi'))
        except Exception:
            return False

    # 1) Verilen yol
    if preferred:
        p = Path(preferred)
        tried.append(str(p))
        if p.exists() and is_valid(p):
            return str(p)

    # 2) Yaygın yollar
    candidates = [
        "/content/Untitled/gpt_qa_checkpoint.pth",
        "/content/gpt_qa_checkpoint.pth",
        "gpt_qa_checkpoint.pth",
        "./gpt_qa_checkpoint.pth",
    ]
    for c in candidates:
        tried.append(c)
        if Path(c).exists() and is_valid(c):
            return c

    # 3) Glob taraması (yalnızca güvenli dizinler)
    search_dirs = ["/content/Untitled", "/content", "."]
    found = []
    for d in search_dirs:
        if not Path(d).exists():
            continue
        for path in glob.glob(os.path.join(d, "**", "*.pth"), recursive=True):
            try:
                st = os.stat(path)
                found.append((st.st_mtime, path))
            except Exception:
                pass
    # En yeni görünenlerden geçerlileri sırayla dene
    for _, path in sorted(found, key=lambda x: x[0], reverse=True):
        tried.append(path)
        if is_valid(path):
            print(f"ℹ️ Checkpoint otomatik bulundu: {path}")
            return path

    raise FileNotFoundError("Checkpoint bulunamadı. Denenen yollar:\n- " + "\n- ".join(tried))

# ---------- Ana akış ----------
def main():
    args = get_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # Checkpoint’i bul
    ckpt_path = find_checkpoint(args.ckpt_path)
    print(f"✅ Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    itos = ckpt['itos']; stoi = ckpt['stoi']
    vocab_size = len(itos)
    sd = ckpt['model_state_dict']

    # Mimariyi checkpoint'ten algıla
    bs_ckpt, n_embd_ckpt, n_layer_ckpt, n_head_ckpt = infer_arch_from_state_dict(sd, args.block_size)
    print(f"🔧 Mimari ⇒ block_size={bs_ckpt}, n_embd={n_embd_ckpt}, n_layer={n_layer_ckpt}, n_head={n_head_ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=bs_ckpt,
        n_embd=n_embd_ckpt,
        n_layer=n_layer_ckpt,
        n_head=n_head_ckpt,
        dropout=0.1
    ).to(device)
    model.load_state_dict(sd)

    # Veri inşası
    defs = parse_input_defs(args.input_path)
    prop_db = parse_props(args.props_path)
    feature2terms = build_feature_index(prop_db)
    qa_examples = build_qa_examples(defs, prop_db, feature2terms)
    print(f"📚 QA örnekleri: {len(qa_examples)}")

    processed = make_dataset(qa_examples, stoi, bs_ckpt)
    train_data, val_data = split_dataset(processed, train_ratio=0.9)
    print(f"▶ Train: {len(train_data)}, ✋ Val: {len(val_data)}")

    # Değerlendirme
    metrics_hist = []
    for ep in range(1, args.eval_passes+1):
        m = evaluate_full(model, val_data, batch_size=args.batch_size, device=device)
        metrics_hist.append(m)
        print(f"[Eval pass {ep}/{args.eval_passes}] loss={m['val_loss']:.4f}  "
              f"tok-acc={m['token_acc']:.2f}%  EM={m['exact_match']:.2f}%  "
              f"(N={m['num_samples']}, labeled={m['num_labeled_tokens']})")

    # CPU Latency
    if len(val_data) == 0:
        print("⚠️ Val kümesi boş; latency testi atlandı.")
        return

    B = min(16, len(val_data))
    xb = torch.stack([val_data[i][0] for i in range(B)])

    def safe_encode(s: str):
        return [stoi.get(c, stoi.get('<pad>', 0)) for c in s]

    prompt = "UMTAS nedir?"
    gen_prompt_ids = torch.tensor(safe_encode(prompt), dtype=torch.long)

    lat = cpu_latency_test(model, xb, gen_prompt_ids, gen_tokens=args.gen_tokens)
    print("\n⏱️ CPU Ölçümleri")
    print(f" - Forward avg time (B={B}, T={bs_ckpt}): {lat['cpu_forward_ms_per_run']:.2f} ms/run")
    print(f" - Forward throughput: {lat['cpu_forward_throughput_tokens_per_s']:.0f} tokens/s")
    print(f" - Generate latency: {lat['cpu_generate_ms_per_token']:.2f} ms/token (tokens={args.gen_tokens})")

    last = metrics_hist[-1]
    print("\n===== ÖZET =====")
    print(f"Val Loss: {last['val_loss']:.4f}")
    print(f"Token Accuracy: {last['token_acc']:.2f}%")
    print(f"Exact Match: {last['exact_match']:.2f}%")
    print(f"Samples: {last['num_samples']}, Labeled tokens: {last['num_labeled_tokens']}")
    print("CPU: forward & generate gecikmeleri yukarıda.")

if __name__ == "__main__":
    main()
