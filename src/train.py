# === train.py (generative: LM pretrain + instruction fine-tune + lightweight retrieval) ===
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from modelroketsan import GPTLanguageModel
import re, unicodedata, random, argparse, math

# ----------------- Ayarlar -----------------
BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='Train a small GPT on ROKETSAN data (PyTorch-only)')
parser.add_argument('--input', type=str, default=str(BASE_DIR / 'input.txt'))
parser.add_argument('--props', type=str, default=str(BASE_DIR / 'veri_properties.txt'))
parser.add_argument('--ckpt', type=str, default=str(BASE_DIR / 'gpt_gen_checkpoint.pth'))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--block_size', type=int, default=0, help='0 or negative = auto from data (95th percentile, capped)')
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--max_iters_lm', type=int, default=2000)
parser.add_argument('--max_iters_inst', type=int, default=3000)
parser.add_argument('--eval_interval', type=int, default=500)
parser.add_argument('--eval_iters', type=int, default=150)
parser.add_argument('--n_embd', type=int, default=512)
parser.add_argument('--n_layer', type=int, default=8)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--warmup_steps', type=int, default=200)
parser.add_argument('--cosine', action='store_true', help='use cosine LR schedule over total steps')
args = parser.parse_args()

input_path      = args.input
props_path      = args.props
checkpoint_path = args.ckpt

batch_size     = args.batch_size
block_size     = args.block_size
learning_rate  = args.learning_rate
max_iters_lm   = args.max_iters_lm
max_iters_inst = args.max_iters_inst
eval_interval  = args.eval_interval
eval_iters     = args.eval_iters
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout        = args.dropout

# ----------------- Yardımcılar -----------------
_TMAP = str.maketrans({'ı':'i','İ':'i','I':'i','ç':'c','Ç':'c','ğ':'g','Ğ':'g','ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'})
def norm(s: str) -> str:
    s = s.strip().casefold().translate(_TMAP)
    s = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))

def canon_name(name: str) -> str:
    return re.sub(r'\s*\(.+?\)\s*', '', name).strip()

def flatten_list(x):
    out = []
    for a in x:
        if isinstance(a, list): out.extend(a)
        else: out.append(a)
    return out

# ----------------- input.txt -> tanımlar -----------------
raw = Path(input_path).read_text(encoding='utf-8').strip()
defs = {}
for blk in raw.split('\n\n'):
    if ':' in blk:
        term, defn = blk.split(':', 1)
        defs[term.strip()] = ' '.join(defn.strip().splitlines())

# ----------------- veri_properties.txt -> özellik verisi -----------------
props_text = Path(props_path).read_text(encoding='utf-8').strip()
prop_db = {}
current = None
for line in props_text.splitlines():
    if not line.strip(): continue
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

# ---------- (opsiyonel) kural tabanlı etiket zenginleştirme (non-breaking) ----------
def add_rule_based_tags(prop_db):
    for full, fields in prop_db.items():
        text_blobs = []
        for k, v in fields.items():
            if isinstance(v, list): text_blobs.extend(v)
            else: text_blobs.append(v)
        blob = " ".join(text_blobs).casefold()

        fields.setdefault('Özellikler', [])

        if any(k in blob for k in ['iha','sıha','siha','helikopter','hava platform','uçaktan','uçak entegrasyonu','hava-yer','hava yer']) \
           and any(k in blob for k in ['kara hedef','tanksavar','yer hedef','kara']):
            fields['Özellikler'].append('Havadan Karaya')

        if ('seyir füzesi' in blob or 'cruise' in blob) and any(k in blob for k in ['kara hedef','uzun menzil']) \
           and any(k in blob for k in ['uçaktan','uçak']):
            fields['Özellikler'].append('Havadan Karaya')

        if any(k in blob for k in ['hava-hava','hava hava','wvr','bvr']):
            fields['Özellikler'].append('Havadan Havaya')

        if any(k in blob for k in ['hava savunma','manpads','alçak irtifa','orta menzil','uzun menzil']) \
           and any(k in blob for k in ['füze','sistem']):
            fields['Özellikler'].append('Karadan Havaya')

        if any(k in blob for k in ['gemisavar','deniz hedef','ciws','torpido','deniz platform']):
            fields['Özellikler'].append('Denizden Denize')

        if any(k in blob for k in ['balistik füze','çok namlulu roketatar','çnra']) or ('balistik' in blob and 'füze' in blob):
            fields['Özellikler'].append('Karadan Karaya')

        if any(k in blob for k in ['omuzdan atılan','tek personel tarafından taşınabilir','manpads']):
            fields['Özellikler'].append('Omuzdan Atılan')

        if 'tanksavar' in blob or 'zırh delici' in blob:
            fields['Özellikler'].append('Tanksavar')

        if any(k in blob for k in ['gemisavar','anti-ship','ashm']):
            fields['Özellikler'].append('Gemisavar')

        fields['Özellikler'] = sorted(set(fields['Özellikler']))

add_rule_based_tags(prop_db)

# ---------- eşanlamlı/ varyant indekslemeleri ----------
_aliases = {
    'havadan karaya': ['hava kara','hava karaya','hava kara','hava-kara','hava-yer','hava yer','air-to-ground','a2g','havadan karaya füze','havadan karaya füzeler'],
    'havadan havaya': ['hava hava','havadan havaya','hava-hava','hava hava','air-to-air','a2a'],
    'karadan havaya': ['kara hava','karadan havaya','karadan-havaya','kara-hava','yerden havaya','surface-to-air','s2a','hava savunma','hava savunma füzesi'],
    'karadan karaya': ['kara kara','karadan karaya','yerden yere','surface-to-surface','s2s','kara-kara'],
    'denizden denize': ['deniz deniz','denizden denize','deniz-deniz','sea-to-sea','gemisavar','anti-ship'],
    'omuzdan atılan': ['omuzdan atilabilen','portable','manpads'],
    'tanksavar': ['tanksavar','atgm','anti-tank','tank savar'],
    'gemisavar': ['gemi savar','anti-ship','ashm'],
}

# ---------- ters indeks (özellik -> terimler) ----------
feature2terms = {}
for full, fields in prop_db.items():
    base = canon_name(full)
    feats = []
    for k,v in fields.items():
        if isinstance(v, list): feats.extend(v)
        else: feats.append(v)
    for f in feats:
        nf = norm(f)
        if not nf: continue
        feature2terms.setdefault(nf, set()).add(base)
        for root, vars_ in _aliases.items():
            if nf == norm(root):
                for var in vars_:
                    feature2terms.setdefault(norm(var), set()).add(base)

# ----------------- Vocab (karakter-seviye) -----------------
def serialize_record(name, fields):
    parts = [f"{canon_name(name)}:"]
    for k, v in fields.items():
        if isinstance(v, list): parts.append(f"{k}: {', '.join(v)}")
        else: parts.append(f"{k}: {v}")
    return " ".join(parts)

plain_corpus = []
for term, definition in defs.items():
    plain_corpus.append(f"{term}: {definition}")
for full, fields in prop_db.items():
    plain_corpus.append(serialize_record(full, fields))

# --- YAMA: instruction şablonlarında geçen tüm olası karakterleri vocab'a dahil et ---
EXTRA_CHARS = " <s></s><pad> Soru: Cevap: Bağlam: ?!.,;:-_()[]/%+*'\"|&—–0123456789"

all_text_for_vocab = "".join(plain_corpus) + EXTRA_CHARS

chars = sorted(set(all_text_for_vocab))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Bu üç özel token zaten tek tek karakter olarak kullanılacağı için
# (örn. '</s>' stringi içindeki '<', '/', '>' karakterleri) 
# ek bir işlem gerekmiyor; ama yine de yoksa ekleyelim:
for tok in ['<s>','</s>','<pad>']:
    if tok not in stoi:
        idx = len(stoi)
        stoi[tok] = idx
        itos[idx] = tok

vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

pad_id = stoi['<pad>']
sep_id = stoi['</s>']

print(f" Vocab boyutu: {vocab_size}")

# ----------------- Instruction metinlerini oluştur (block_size otomasyonu için) -----------------
def fields_to_sentences(name, fields):
    kv = []
    for k, v in fields.items():
        if isinstance(v, list): kv.append(f"{k}: {', '.join(v)}")
        else: kv.append(f"{k}: {v}")
    random.shuffle(kv)
    return f"{canon_name(name)} için bilgiler: " + " | ".join(kv)

def build_instruction_texts():
    data = []
    # 1) Tanım soruları (çeşitli kalıplar)
    def_qv = [
        "nedir?", "ne demektir?", "ne anlama gelir?", "kısaca açıkla.",
        "hakkında bilgi ver.", "tanımını yap.", "özetle açıkla.",
        "hakkında kısa bilgi.", "nedir acaba?",
    ]
    for term, definition in defs.items():
        for _ in range(2):  # her terim için iki varyant üret
            q = f"{term} {random.choice(def_qv)}"
            # Bağlam: sadece tanım ya da tanıma + ilgili öz nitelikler (varsa)
            ctx = definition
            # Yakın kayıtları bağlama ekle (zengin bağlam öğrenimi)
            extra = []
            for full, fields in prop_db.items():
                if canon_name(full) == canon_name(term):
                    extra.append(fields_to_sentences(full, fields))
                    break
            if extra:
                ctx = ctx + " | " + " | ".join(extra)
            a = definition
            data.append( "Soru: " + q + f" {itos[sep_id]} Bağlam: " + ctx + f" {itos[sep_id]} Cevap: " + a )

    # 2) Özellik soruları (parafrazlı)
    feat_qv = [
        "özellikleri nelerdir?", "hangi özelliklere sahip?", "özeti nedir?",
        "temel kabiliyetleri neler?", "detaylı özelliklerini açıkla.",
        "hangi kabiliyetleri var?", "hangi alanlarda kullanılır?",
    ]
    for full, fields in prop_db.items():
        base = canon_name(full)
        ctx = fields_to_sentences(full, fields)
        a_parts = []
        for k, v in fields.items():
            if isinstance(v, list): a_parts.append(f"{k}: {', '.join(v)}")
            else: a_parts.append(f"{k}: {v}")
        a = f"{base} — " + "; ".join(a_parts) if a_parts else base
        for _ in range(2):
            q = f"{base} {random.choice(feat_qv)}"
            data.append( "Soru: " + q + f" {itos[sep_id]} Bağlam: " + ctx + f" {itos[sep_id]} Cevap: " + a )

    # 3) Özellikten listeleme soruları + sayım (kategori bazlı)
    count_qv = [
        "kaç tane var?", "sayısı kaç?", "toplam kaç adet?", "adet olarak kaç?",
    ]
    list_qv = [
        "hangileri?", "listeler misin?", "örnekleri?", "içeren sistemler hangileri?",
    ]
    for nf, terms in feature2terms.items():
        if not terms:
            continue
        # orijinal yazımı yakala
        any_original = None
        for full, fields in prop_db.items():
            feats = []
            for k,v in fields.items():
                feats.extend(v if isinstance(v, list) else [v])
            for f in feats:
                if norm(f) == nf:
                    any_original = f; break
            if any_original: break
        feat_str = any_original or nf
        ctx = "Aşağıdaki sistemler ilgili özelliği taşır: " + ", ".join(sorted(terms))
        # Liste soruları
        for _ in range(2):
            q = f"{feat_str} {random.choice(list_qv)}"
            a = ", ".join(sorted(terms))
            data.append( "Soru: " + q + f" {itos[sep_id]} Bağlam: " + ctx + f" {itos[sep_id]} Cevap: " + a )
        # Sayım soruları
        for _ in range(2):
            q = f"{feat_str} {random.choice(count_qv)}"
            a = f"Toplam: {len(set(terms))}"
            data.append( "Soru: " + q + f" {itos[sep_id]} Bağlam: " + ctx + f" {itos[sep_id]} Cevap: " + a )

    random.shuffle(data)
    return data

inst_texts = build_instruction_texts()

# ----------------- block_size otomatik belirleme -----------------
if block_size is None or block_size <= 0:
    all_texts = ["<s>" + t for t in (plain_corpus + inst_texts)]
    lengths = sorted(len(t) for t in all_texts)
    if lengths:
        idx = max(0, int(0.95 * (len(lengths) - 1)))
        auto_bs = min(768, max(128, lengths[idx] + 8))
        block_size = auto_bs
    else:
        block_size = 256
print(f" block_size = {block_size}")


# ----------------- Dataset yardımcıları -----------------
def make_causal_example(text):
    ids = encode(text)
    if len(ids) >= block_size:
        ids = ids[-block_size:]
    # Causal LM: giriş = hedef = aynı, yalnızca padding -100
    x = [pad_id] * (block_size - len(ids)) + ids
    y = [-100] * (block_size - len(ids)) + ids
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 1) LM ön-eğitim seti (serbest metin)
lm_examples = [ make_causal_example(t) for t in plain_corpus ]

# 2) Instruction seti (Soru + Bağlam + Cevap -> SFT: sadece cevap kısmına kayıp)
def make_sft_example(full_text: str, answer_key: str = 'Cevap:'):
    # answer_only loss: 'answer_key' metninden ÖNCEKİ tüm tokenlar -100
    ids = encode(full_text)
    # pozisyonu karakter-düzeyi ile bul
    try:
        ans_char_pos = full_text.index(answer_key) + len(answer_key)
    except ValueError:
        ans_char_pos = 0
    ans_tok_pos = len(encode(full_text[:ans_char_pos]))
    # block kırpma
    if len(ids) >= block_size:
        # kırpma durumunda cevap bölgesi sona yakın kalmalı; son block'u alıyoruz
        start = len(ids) - block_size
        ans_tok_pos = max(0, ans_tok_pos - start)
        ids = ids[start:]
    x = [pad_id] * (block_size - len(ids)) + ids
    y = [-100] * len(x)
    # hedefleri doldur: sadece cevap kısmı
    for i in range(block_size - len(ids) + ans_tok_pos, block_size):
        y[i] = x[i]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_instruction_pairs():
    return [ make_sft_example(t) for t in inst_texts ]

inst_examples = build_instruction_pairs()

print(f" LM örnekleri: {len(lm_examples)} | Instruction örnekleri: {len(inst_examples)}")

# ----------------- Batching -----------------
def get_batch(examples):
    ix = torch.randint(0, len(examples), (batch_size,))
    xb = torch.stack([examples[i][0] for i in ix.tolist()]).to(device)
    yb = torch.stack([examples[i][1] for i in ix.tolist()]).to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss(model, examples):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(examples)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    out['loss'] = losses.mean()
    model.train()
    return out

# ----------------- Model -----------------
model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

# LR scheduler (warmup + optional cosine)
total_steps = max_iters_lm + max_iters_inst
def lr_lambda(step):
    if step < args.warmup_steps:
        return float(step + 1) / float(max(1, args.warmup_steps))
    if args.cosine:
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------- 1) LM Ön-eğitim -----------------
for it in range(max_iters_lm):
    if it % eval_interval == 0 or it == max_iters_lm - 1:
        losses = estimate_loss(model, lm_examples)
        print(f"[LM] step {it}: loss {losses['loss']:.4f}")
    xb, yb = get_batch(lm_examples)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()

# ----------------- 2) Instruction İnce Ayar -----------------
for it in range(max_iters_inst):
    if it % eval_interval == 0 or it == max_iters_inst - 1:
        losses = estimate_loss(model, inst_examples)
        print(f"[INST] step {it}: loss {losses['loss']:.4f}")
    xb, yb = get_batch(inst_examples)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()

# ----------------- Kaydet (son ve en iyi) -----------------
config = {
    'block_size': block_size,
    'n_embd': args.n_embd,
    'n_layer': args.n_layer,
    'n_head': args.n_head,
    'dropout': dropout,
    'pad_token': '<pad>',
    'sep_token': '</s>',
}
torch.save({
    'model_state_dict': model.state_dict(),
    'itos': itos,
    'stoi': stoi,
    'config': config,
}, checkpoint_path)
print(f" Model kaydedildi: {checkpoint_path}")

# ----------------- Basit Inference + Retrieval -----------------
@torch.no_grad()
def top_k_records(query, k=6):
    nq = norm(query)
    # 1) Doğrudan özellik eşleşmesi
    hits = list(feature2terms.get(nq, []))
    # 2) query içindeki kelimeleri parçala ve basit genişletme yap
    toks = [t for t in re.split(r'[\s,;:]+', nq) if t]
    # Özellik alias'larını da deneyelim
    for root, vars_ in _aliases.items():
        if any(norm(v) in nq for v in vars_) or norm(root) in nq:
            hits.extend(list(feature2terms.get(norm(root), [])))
            for v in vars_:
                hits.extend(list(feature2terms.get(norm(v), [])))
    # 3) terim adlarında kaba eşleşme
    for name in prop_db.keys():
        if any(t in norm(name) for t in toks):
            hits.append(canon_name(name))
    # benzersiz + skor yerine frekans sezgisi
    uniq = []
    for h in hits:
        if h not in uniq: uniq.append(h)
    return uniq[:k]

def build_context(cands):
    lines = []
    for full, fields in prop_db.items():
        base = canon_name(full)
        if base in cands:
            lines.append(serialize_record(full, fields))
    return " | ".join(lines) if lines else ""

def generate_answer(prompt, max_new_tokens=180, temperature=0.9, top_k=0):
    model.eval()
    start = "<s>" + prompt
    idx = torch.tensor([encode(start)], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            # basit durdurma: Cevap: ... satırı bitti varsayımı
            if itos[next_token.item()] == '\n':
                pass
    txt = decode(idx[0].tolist())
    return txt[len("<s>"):]

def answer(query):
    cands = top_k_records(query, k=8)
    ctx = build_context(cands)
    prompt = f"Soru: {query} {itos[sep_id]} Bağlam: {ctx} {itos[sep_id]} Cevap: "
    return generate_answer(prompt)

# Örnek kullanım (eğitim sonunda hızlı duman testi)
print("\n Örnek:")
print("Soru: havadan karaya füzeler hangileri?")
print("Cevap:", answer("havadan karaya füzeler hangileri?")[:400])
