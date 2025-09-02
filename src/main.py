
import re, sys, unicodedata, json, os, datetime
from pathlib import Path
import torch
from modelroketsan import GPTLanguageModel

RAW_PATH       = '/content/input.txt'
PROP_PATH      = '/content/veri_properties.txt'
HISTORY_PATH   = '/content/chat_history.jsonl'
EXPORT_MD_PATH = '/content/chat_history.md'

# Davranış ayarları
STRICT_DEFINITION = True   # "nedir?" türü sorularda tanım yoksa özelliklere düşme

# --- Veri yükleme (input.txt) ---
raw = Path(RAW_PATH).read_text(encoding='utf-8').strip()

# Orijinal terim -> tanım sözlüğü
d = {}
for blk in re.split(r'\n{2,}', raw):
    if ':' in blk:
        term, defn = blk.split(':', 1)
        d[term.strip()] = ' '.join(defn.strip().splitlines())

# --- Normalizasyon yardımcıları ---
_TMAP = str.maketrans({
    'ı':'i','İ':'i','I':'i','ç':'c','Ç':'c','ğ':'g','Ğ':'g',
    'ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'
})
def norm(s: str) -> str:
    s = s.strip().casefold().translate(_TMAP)
    s = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))

def norm_squash(s: str) -> str:
    s = norm(s)
    return re.sub(r'[^0-9a-z]+', '', s)

def canon_name(name: str) -> str:
    return re.sub(r'\s*\(.+?\)\s*', '', name).strip()

# --- Alias çıkarımı: başlıktan varyantlar üret ---
def make_aliases(term: str):
    aliases = set()
    term_clean = re.sub(r'\s+', ' ', term or '').strip()
    if term_clean:
        aliases.add(term_clean)

    m = re.search(r'^(.*?)\s*\(([^()]*)\)\s*$', term_clean)
    if m:
        a1 = m.group(1).strip()
        a2 = m.group(2).strip()
        if a1: aliases.add(a1)
        if a2: aliases.add(a2)

    for sep in ['/', '-', '–', '—', ':', '_']:
        if sep in term_clean:
            for p in (x.strip() for x in term_clean.split(sep)):
                if len(p) >= 2:
                    aliases.add(p)

    spaced = re.sub(r'[-–—/_]+', ' ', term_clean).strip()
    if spaced:
        aliases.add(spaced)

    squashed = re.sub(r'[\s\-–—/_:()]+', '', term_clean)
    if squashed:
        aliases.add(squashed)

    DROP = {'a','o','u','i','of','and','the'}
    aliases = {a for a in aliases if len(norm(a)) >= 2 and norm(a) not in DROP}
    return aliases

# --- veri_properties.txt oku ---
props = {}        # adı -> { Alan:..., Tür:..., Menşei:..., Özellikler:..., ... }
props_text = {}   # adı -> tek satır özet (arama için)
# Oturum içi öğrenilen özellikler (resmi kaynağın üstüne yazmaz; sadece eksik yerleri tamamlar)
learned_props = {}  # adı -> { Alan:..., Tür:..., Menşei:..., Özellikler:..., ... }

if os.path.exists(PROP_PATH):
    ptxt = Path(PROP_PATH).read_text(encoding='utf-8').strip()
    blocks, cur = [], []
    for line in ptxt.splitlines():
        if line.strip().endswith('):') and '(' in line:
            if cur:
                blocks.append('\n'.join(cur).strip())
                cur = []
        cur.append(line)
    if cur:
        blocks.append('\n'.join(cur).strip())

    def flush_current(k, buf, rec):
        if k is not None:
            rec[k] = ' '.join(' '.join(buf).split())
        return None, []

    for blk in blocks:
        lines = [l.rstrip() for l in blk.splitlines() if l.strip()]
        head = lines[0]
        name = head[:-1].strip()
        rec = {}
        k = None
        buf = []
        for l in lines[1:]:
            m = re.match(r'^([A-Za-zÇĞİÖŞÜçğıöşü\s/()-]+):\s*(.*)$', l)
            if m:
                k, buf = flush_current(k, buf, rec)
                k = m.group(1).strip()
                val = m.group(2).strip()
                buf = [val] if val else []
            else:
                buf.append(l.strip())
        k, buf = flush_current(k, buf, rec)
        props[name] = rec
        props_text[name] = '; '.join(f"{kk}: {vv}" for kk, vv in rec.items())

# === PATCH: Özellik etiketleri + alias + feature->terms indeksi ===
def split_features(val):
    if not val:
        return []
    if isinstance(val, list):
        items = val
    else:
        items = re.split(r',|;|\n|\||/|•', val)
    return [i.strip(' .;') for i in items if i and i.strip()]

def add_rule_based_tags_inplace(props_dict):
    for name, rec in props_dict.items():
        # Türkçe karakter/aksan normalizasyonuyla tek gövde
        blob_parts = []
        for k, v in rec.items():
            blob_parts.append(str(v))
        blob = norm(" ".join(blob_parts))

        feats = set(split_features(rec.get('Özellikler', '')))

        # ---- A2G (Havadan Karaya): hava platform + kara hedef; hava savunma sistemlerini hariç tut
        is_air_defense = any(k in blob for k in [
            'hava savunma','ciws','korkut','tehditlere karsi','tehditlerine karsi'
        ])
        a2g_platform = any(k in blob for k in [
            'ucaktan atilan','hava platform','helikopter','iha','siha',
            'ucak entegrasyonu','helikopter entegrasyonu','iha/siha uyumu','iha siha uyumu'
        ])
        a2g_ground = any(k in blob for k in ['kara hedef','yer hedef','tanksavar'])
        if a2g_platform and a2g_ground and not is_air_defense:
            feats.add('Havadan Karaya')

        # A2A (Havadan Havaya)
        if any(k in blob for k in ['hava-hava','hava hava','wvr','bvr']):
            feats.add('Havadan Havaya')

        # S2A (Karadan Havaya) - hava savunma/manpads/HİSAR vb.
        if any(k in blob for k in ['hava savunma','manpads','alcak irtifa','orta menzil','uzun menzil']) \
           and any(k in blob for k in ['fuze','sistem']):
            feats.add('Karadan Havaya')

        # S2S (Karadan Karaya): balistik, ÇNRA vb.
        if any(k in blob for k in ['balistik fuze','cok namlulu roketatar','cnra']) or ('balistik' in blob and 'fuze' in blob):
            feats.add('Karadan Karaya')

        # Denizden Denize
        if any(k in blob for k in ['gemisavar','deniz hedef','ciws','torpido','deniz platform']):
            feats.add('Denizden Denize')

        # Omuzdan Atılan
        if any(k in blob for k in ['omuzdan atilan','tek personel tarafindan tasinabilir','manpads']):
            feats.add('Omuzdan Atılan')

        # Tanksavar
        if 'tanksavar' in blob or 'zirh delici' in blob:
            feats.add('Tanksavar')

        # Gemisavar etiketi
        if any(k in blob for k in ['gemisavar','anti-ship','ashm']):
            feats.add('Gemisavar')

        if feats:
            rec['Özellikler'] = ', '.join(sorted(feats))

add_rule_based_tags_inplace(props)

FEATURE_ALIASES = {
    'havadan karaya': ['hava-yer','hava yer','air-to-ground','a2g','havadan karaya füze','havadan karaya füzeler'],
    'havadan havaya': ['hava-hava','hava hava','air-to-air','a2a'],
    'karadan havaya': ['yerden havaya','surface-to-air','s2a','hava savunma','hava savunma füzesi'],
    'karadan karaya': ['yerden yere','surface-to-surface','s2s','kara-kara'],
    'denizden denize': ['sea-to-sea','gemisavar','anti-ship'],
    'omuzdan atılan': ['omuzdan atilabilen','portable','manpads'],
    'tanksavar': ['atgm','anti-tank','tank savar'],
    'gemisavar': ['anti-ship','ashm'],
}

FEATURE_DEFINITIONS = {
    'havadan karaya' : "Hava platformlarından kara/yer hedeflerine atılan güdümlü mühimmat ailesi (A2G).",
    'havadan havaya' : "Hava platformlarından hava hedeflerine karşı kullanılan güdümlü füzeler (A2A).",
    'karadan havaya' : "Yer konuşlu hava savunma sistemleri; hava hedeflerini önler (S2A).",
    'karadan karaya' : "Yer konuşlu sistemlerden yine yer hedeflerine atılan roket/füzeler (S2S).",
    'denizden denize' : "Deniz platformlarından deniz hedeflerine (gemi/denizaltı) karşı silahlar.",
    'omuzdan atılan' : "Tek personelce taşınıp atılabilen (MANPADS/ATGM) hafif güdümlü silahlar.",
    'tanksavar'      : "Zırhlı hedefleri imha etmek için tasarlanmış güdümlü mühimmat ailesi.",
    'gemisavar'      : "Deniz hedeflerine karşı kullanılan, çoğunlukla aktif radar/IR güdümlü füzeler.",
}

# Özellik -> terimler (alias'larla birlikte)
feature2terms = {}
for name, rec in props.items():
    feats = split_features(rec.get('Özellikler', ''))
    base = canon_name(name)
    for f in feats:
        nf = norm(f)
        if not nf:
            continue
        feature2terms.setdefault(nf, set()).add(base)
        for root, vars_ in FEATURE_ALIASES.items():
            if nf == norm(root):
                for var in vars_:
                    feature2terms.setdefault(norm(var), set()).add(base)

# --- Eşleşme dizinleri (tanım + özellik başlıklarından) ---
all_terms = sorted(set(list(d.keys()) + list(props.keys())))
alias2term = {}
alias2term_squash = {}
for term in all_terms:
    for a in make_aliases(term):
        alias2term[norm(a)] = term
        alias2term_squash[norm_squash(a)] = term

ALIAS_NORM_DESC   = sorted(alias2term.keys(), key=len, reverse=True)
ALIAS_SQUASH_DESC = sorted(alias2term_squash.keys(), key=len, reverse=True)

# === Dinamik öğrenme yardımcıları (öğretme) ===
def rebuild_alias_indexes():
    global all_terms, alias2term, alias2term_squash, ALIAS_NORM_DESC, ALIAS_SQUASH_DESC
    all_terms = sorted(set(list(d.keys()) + list(props.keys())))
    alias2term = {}
    alias2term_squash = {}
    for term in all_terms:
        for a in make_aliases(term):
            alias2term[norm(a)] = term
            alias2term_squash[norm_squash(a)] = term
    ALIAS_NORM_DESC   = sorted(alias2term.keys(), key=len, reverse=True)
    ALIAS_SQUASH_DESC = sorted(alias2term_squash.keys(), key=len, reverse=True)

def persist_new_definition(term: str, definition: str):
    block = f"\n\n{term}: {definition}\n"
    # 1) Öncelik: RAW_PATH
    try:
        with open(RAW_PATH, 'a', encoding='utf-8') as f:
            f.write(block)
        return True, RAW_PATH
    except Exception:
        pass
    # 2) Alternatif: script ile aynı klasördeki input.txt
    try:
        local = Path(__file__).resolve().parent / 'input.txt'
        with open(local, 'a', encoding='utf-8') as f:
            f.write(block)
        return True, str(local)
    except Exception:
        return False, None

def teach_definition_inline(text: str):
    # Beklenen biçim: "TERİM: <Tanım veya Alan: Değer; Özellikler: a,b; ...>"
    if ':' not in text:
        return False, "Biçim: /ogret Terim: Tanım | veya /ogret Terim: Tür: X; Menşei: Y; Özellikler: a, b"
    term, payload = text.split(':', 1)
    term = term.strip()
    payload = payload.strip()
    if not term or not payload:
        return False, "Biçim: /ogret Terim: Tanım"
    # Alan anahtarları var mı?
    has_field = any(k in payload for k in ['Tür:', 'Menşei:', 'Özellikler:', 'Ozellikler:', 'Tanım:', 'Tanim:'])
    if has_field:
        # satır/; ayrımıyla parçala
        kvs = re.split(r';|\n', payload)
        for kv in kvs:
            if ':' not in kv:
                continue
            k, v = kv.split(':', 1)
            k = k.strip()
            v = v.strip()
            key_norm = k.lower()
            if key_norm in ('tanım', 'tanim'):
                d[term] = v
                ok, _ = persist_new_definition(term, v)
            elif key_norm in ('özellikler', 'ozellikler'):
                set_user_field(term, 'Özellikler', [s.strip() for s in re.split(r',|\|', v) if s.strip()])
            elif key_norm == 'tür':
                set_user_field(term, 'Tür', v)
            elif key_norm in ('menşei', 'mensei', 'menşe', 'koken', 'köken', 'origin'):
                set_user_field(term, 'Menşei', v)
        rebuild_alias_indexes()
        return True, f" Öğrenildi: {term} (oturum geçerli; tanım varsa dosyaya eklendi)"
    else:
        # Sade tanım
        d[term] = payload
        rebuild_alias_indexes()
        ok, path = persist_new_definition(term, payload)
        if ok:
            return True, f" Öğrenildi ve kaydedildi: {term} (dosya: {path})"
        else:
            return True, f" Öğrenildi (oturum içi). Kaydetme başarısız: {term}"

def summarize_types(feature_filter_root: str | None = None):
    # Tür alanlarını topla; feature filtre varsa yalnız o özelliğe sahip kayıtlar
    types = []
    for name, rec in props.items():
        if feature_filter_root:
            feats = split_features(rec.get('Özellikler', ''))
            keys = [norm(feature_filter_root)] + [norm(s) for s in FEATURE_ALIASES.get(feature_filter_root, [])]
            if not any(norm(f) in keys for f in feats):
                continue
        t = rec.get('Tür')
        if t:
            types.append(t.strip())
    # benzersiz, tutarlı sıra
    uniq = []
    for t in types:
        if t not in uniq:
            uniq.append(t)
    lines = [f"Tür Sayısı: {len(uniq)}", "Türler:"]
    for i, t in enumerate(uniq, 1):
        lines.append(f"{i}. {t}")
    return "\n".join(lines)

# --- Soru kalıpları ve noktalama ---
QUESTION_TAIL_RE = re.compile(
    r'\b('
    r'nedir|ne|ne demek|ne anlama gelir|kimdir|neredir|nedir acaba|'
    r'tanimi|tanımı|aciklamasi|açıklaması|hakkinda bilgi|hakkında bilgi'
    r')\b\??',
    flags=re.IGNORECASE
)
PUNCT_RE = re.compile(r'[?!.,;:…“”"\'`´^~•·()$begin:math:display$$end:math:display${}<>]+')

# --- Alan/niyet sözlüğü ---
FIELD_SYNS = {
    'Özellikler': ['ozellik','ozellikleri','özellik','özellikleri','ozellikler','özellikler','ozelligi','özelliği','detay','kabiliyet','kabiliyetleri','tumu','hepsi'],
    'Tür'       : ['tur','turu','tür','türü','kategori','type','sinif','sınıf'],
    'Menşei'    : ['mensei','menşei','menşe','origin','koken','köken'],
    'Kuruluş'   : ['kurulus','kuruluş','yil','yılı','kurulma','kurulus yili','yıl'],
    'Alan'      : ['alan','faaliyet alan','sektor','sektör','uzmanlik','uzmanlık','alanı'],
    'Kullanım'  : ['kullanim','kullanım','kullanim amaci','kullanim alan','rol','rolu','rolü'],
    'Üretici'   : ['uretici','üretici','ureten','manufacturer','yapan'],
}
LIST_HINTS  = ['hangileri','kimlerde','kimler','olanlar','içeren','içerenler','sahip olanlar','listesi','hangi','hepsi']
COUNT_HINTS = ['kaç','kac','kaç tane','kac tane','sayisi','sayısı','adet','sayi','sayı','how many','count','number']
TYPE_HINTS  = ['tür','tur','türler','türleri','turler','turleri']

# --- Sınıflandırma (küresel) ipuçları ---
CLASSIFY_HINTS = [
    'siniflandir','sınıflandır','siniflandirma','sınıflandırma',
    'gruplandir','gruplandır','gruplama',
    'kategoriye gore','kategoriye göre','kategori bazli','kategori bazlı',
    'ozelliklere gore','özelliklere gore','özelliklere göre','ozellik bazli','özellik bazlı',
    'etiketlere gore','etiketlere göre','kategori dagilimi','kategori dağılımı','dagilim','dağılım'
]

def wants_count(q: str) -> bool:
    n = norm(q)
    return any(norm(h) in n for h in COUNT_HINTS)

def wants_type_group(q: str) -> bool:
    n = norm(q)
    return any(norm(h) in n for h in TYPE_HINTS) and (wants_count(q) or any(h in n for h in ['türler','turler','türleri','turleri']))

# --- Coreference patch: önceki terime gönderme tespiti ---
COREF_PRONOUNS = r'\b(bu|şu|o|bunlar|şunlar|onlar|bunun|şunun|onun|buna|şuna|ona|bundan|şundan|ondan)\b'
def is_coref_like(q: str) -> bool:
    n = norm(q)
    # Zamir / işaret sözcükleri
    if re.search(COREF_PRONOUNS, ' ' + q.lower() + ' '):
        return True
    # Alan/niyet kelimeleri veya liste/sayım veya tanım kalıbı
    field_hit = any(norm(s) in n for syns in FIELD_SYNS.values() for s in syns) or any(
        k in n for k in ['özellik','ozellik','tür','tur','menşei','mensei'])
    list_or_count = wants_count(q) or any(norm(h) in n for h in LIST_HINTS)
    return field_hit or list_or_count or is_definition_query(q)

# --- Çoklu terim çıkarımı (aday listesi) ---
def extract_terms_raw(q: str):
    """Sorgudaki TÜM terimleri (d/props başlıkları) döndür (tekrarları at)."""
    n_keep = norm(q)
    n_sq   = norm_squash(q)
    found = []
    seen = set()
    for key in ALIAS_NORM_DESC:
        if key and key in n_keep:
            t = alias2term[key]
            if t not in seen:
                seen.add(t); found.append(t)
    for key in ALIAS_SQUASH_DESC:
        if key and key in n_sq:
            t = alias2term_squash[key]
            if t not in seen:
                seen.add(t); found.append(t)
    if found:
        return found
    # --- Fuzzy eşleşme (ör. MAM-L ~ maml) ---
    def _lev(a: str, b: str, max_d: int = 1) -> int:
        # küçük ve hızlı Levenshtein (erken kesme)
        if abs(len(a) - len(b)) > max_d:
            return max_d + 1
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev = dp[0]
            dp[0] = i
            best = dp[0]
            for j, cb in enumerate(b, 1):
                cur = dp[j]
                cost = 0 if ca == cb else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
                if dp[j] < best: best = dp[j]
            if best > max_d:
                return max_d + 1
        return dp[-1]
    toks = [t for t in re.split(r'\s+', n_sq) if len(t) >= 3]
    for tok in toks:
        # alias2term_squash üzerinden yakın eşleşme ara
        for key, term in alias2term_squash.items():
            if len(key) >= 3 and _lev(tok, key, max_d=1) <= 1:
                if term not in seen:
                    seen.add(term); found.append(term)
        if found:
            break
    # --- Coreference fallback: hiç terim bulunamadıysa önceki konuya bağlan ---
    if not found and is_coref_like(q):
        last = get_last_subject()
        if last:
            return [last]
    return found

# Tanım sinyali & kategori tespiti
DEF_HINTS = ['nedir','ne demek','ne anlama gelir','kimdir','neredir','tanimi','tanımı','aciklamasi','açıklaması','kisaca acikla','kısaca açıkla',' ne ']
def is_definition_query(q: str) -> bool:
    n = f" {norm(q)} "
    if ' ne ' in n:
        terms = extract_terms_raw(q)
        if terms:
            return True
    return any(f" {h} " in n for h in DEF_HINTS if h != ' ne ') or (' nedir ' in n)

def extract_feature_from_query(q: str):
    n = norm(q)
    for root, syns in FEATURE_ALIASES.items():
        keys = [norm(root)] + [norm(s) for s in syns]
        if any(k in n for k in keys):
            return root
    return None

# --- Sorgudan tek terim çıkar (tek varlık gerektiren yanıtlar için) ---
def extract_term(q: str):
    terms = extract_terms_raw(q)
    if len(terms) == 1:
        return terms[0]
    return None  # 0 veya 2+

# --- Gruplama yardımcıları (özellik köklerine göre) ---
def terms_for_feature_root(root: str):
    keys = [norm(root)] + [norm(s) for s in FEATURE_ALIASES.get(root, [])]
    items = set()
    for k in keys:
        items |= feature2terms.get(k, set())
    return sorted(items)

def summarize_feature_counts(include_lists: bool):
    lines = []
    total_unique = set()
    roots = list(FEATURE_ALIASES.keys())
    for r in roots:
        items = terms_for_feature_root(r)
        total_unique |= set(items)
        if include_lists and items:
            lines.append(f" - {r.title()}: {len(items)}\n   " + ", ".join(items))
        else:
            lines.append(f" - {r.title()}: {len(items)}")
    lines.append(f"\n Genel toplam (benzersiz öğe): {len(total_unique)}")
    return "\n".join(lines)

# --- Niyet tespiti (ÖNCE sınıflandırma, sonra özellik/alan > tanım > kategori/list) ---
def detect_intent(q: str):
    n = norm(q)

    # 0) Özelliklere göre sınıflandırma / dağılım talebi
    if any(h in n for h in CLASSIFY_HINTS):
        return ('classify', None)

    # 1) Açık özellik/alan sinyali
    for s in FIELD_SYNS['Özellikler']:
        if norm(s) in n:
            return ('features', None)
    for field, syns in FIELD_SYNS.items():
        if field == 'Özellikler':
            continue
        for s in syns + [field]:
            if norm(s) in n:
                return ('field', field)

    # 1.5) Tür gruplama/sayım
    if wants_type_group(q):
        return ('type_group', None)

    # 2) Tanım sinyali (ne/nedir ...)
    if is_definition_query(q):
        return ('definition', None)

    # 3) Kategori bazlı liste (COUNT veya LIST varsa ya da def değilse)
    cat_root = extract_feature_from_query(q)
    if cat_root:
        if any(h in n for h in LIST_HINTS) or any(h in n for h in COUNT_HINTS) or not is_definition_query(q):
            return ('feature_list', cat_root)

    # 4) Genel listeleme
    if any(h in n for h in LIST_HINTS):
        return ('list', None)

    # 5) Varsayılan: tanım
    return ('definition', None)

# --- Özellik yardımcıları ---
def props_for_entity(name: str):
    if name in props:
        rec = props[name]
        s = '; '.join(f"{k}: {v}" for k, v in rec.items()) if rec else "(kayıt yok)"
        return name, s
    n = norm(name); nsq = norm_squash(name)
    if n in alias2term:
        t = alias2term[n]
        if t in props:
            rec = props[t]
            s = '; '.join(f"{k}: {v}" for k, v in rec.items()) if rec else "(kayıt yok)"
            return t, s
    if nsq in alias2term_squash:
        t = alias2term_squash[nsq]
        if t in props:
            rec = props[t]
            s = '; '.join(f"{k}: {v}" for k, v in rec.items()) if rec else "(kayıt yok)"
            return t, s
    return None, None

def get_field_value(entity: str, field: str):
    """Resmi kaynaktan alan değeri, yoksa oturum içi öğrenilen değeri döndür.
    Çakışma durumunda resmi değer önceliklidir; not olarak kullanıcı bilgisini bildirir.
    Returns: (value, source, conflict_note)
    """
    source = None
    conflict = None
    # entity canonical çözümle
    if entity not in props:
        n = norm(entity); nsq = norm_squash(entity)
        if n in alias2term and alias2term[n] in props:
            entity = alias2term[n]
        elif nsq in alias2term_squash and alias2term_squash[nsq] in props:
            entity = alias2term_squash[nsq]
    # 1) Resmi
    if entity in props:
        if field in props[entity]:
            return props[entity][field], 'official', None
        tgt = norm(field)
        for k, v in props[entity].items():
            if norm(k) == tgt:
                return v, 'official', None
    # 2) Kullanıcı bilgisi
    lp = learned_props.get(entity, {})
    if field in lp:
        # Çakışmayı tespit et (resmi varsa ve farklıysa)
        if entity in props and field in props[entity]:
            if props[entity][field] != lp[field]:
                conflict = f"(Resmi kaynak öncelikli; kullanıcı bilgisi farklı: {lp[field]})"
        return lp[field], 'user', conflict
    tgt = norm(field)
    for k, v in lp.items():
        if norm(k) == tgt:
            if entity in props and k in props[entity] and props[entity][k] != v:
                conflict = f"(Resmi kaynak öncelikli; kullanıcı bilgisi farklı: {v})"
            return v, 'user', conflict
    return None, None, None

def get_features_list(entity: str):
    """Sadece Özellikler alanını liste olarak döndür. Resmi yoksa kullanıcı bilgisini kullan."""
    feats_off = None
    feats_user = None
    en = entity
    if en not in props:
        n = norm(en); nsq = norm_squash(en)
        if n in alias2term and alias2term[n] in props:
            en = alias2term[n]
        elif nsq in alias2term_squash and alias2term_squash[nsq] in props:
            en = alias2term_squash[nsq]
    if en in props:
        v = props[en].get('Özellikler')
        if isinstance(v, str):
            feats_off = [s.strip() for s in re.split(r',|;|\n|\||/|•', v) if s.strip()]
        elif isinstance(v, list):
            feats_off = v
    lp = learned_props.get(en, {})
    v2 = lp.get('Özellikler')
    if v2:
        if isinstance(v2, str):
            feats_user = [s.strip() for s in re.split(r',|;|\n|\||/|•', v2) if s.strip()]
        elif isinstance(v2, list):
            feats_user = v2
    return feats_off, feats_user

def set_user_field(entity: str, field: str, value):
    en = entity
    if en not in learned_props:
        learned_props[en] = {}
    learned_props[en][field] = value
    # Özellikler alanı güncellendiyse kategoriler için indeksleri tazelemek gerekli olabilir (hafif)
    # Burada sadece alias indeksini tazelemek yeterli
    rebuild_alias_indexes()

def list_by_keyword(q: str):
    n = norm(q)
    t = extract_term(q)
    if t:
        n = n.replace(norm(t), ' ')
    stop = set(['hangileri','kimlerde','kimler','olanlar','olan','var','mi','mı','mu','mü',
                'listesi','hangi','hepsi','özellik','ozellik','ozellikler','özellikler','tür','tur','menşei','mensei'])
    kws = [w for w in re.split(r'\s+', n) if w and len(w) > 2 and w not in stop]
    if not kws:
        return []
    out = []
    for name, rec in props.items():
        hay = norm(name) + ' ' + norm(' '.join(rec.keys())) + ' ' + norm(' '.join(rec.values()))
        if all(kw in hay for kw in kws):
            out.append(name)
    return sorted(set(out), key=str.lower)

def list_by_feature(q: str):
    n = norm(q)
    hits = set()
    for root, syns in FEATURE_ALIASES.items():
        keys = [norm(root)] + [norm(s) for s in syns]
        if any(k in n for k in keys):
            for k in keys:
                hits |= feature2terms.get(k, set())
    return sorted(hits)

# === Geçmiş (memory + dosya) ===
history = []
save_enabled = True

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass

# --- Coreference patch: geçerli konuyu takip et ---
CURRENT_SUBJECT = None  # en son tekil çözümlenen terim (canon)
def set_current_subject(t: str | None):
    global CURRENT_SUBJECT
    if t:
        CURRENT_SUBJECT = canon_name(t)
def get_last_subject() -> str | None:
    if CURRENT_SUBJECT:
        return CURRENT_SUBJECT
    for rec in reversed(history):
        t = rec.get("term")
        if isinstance(t, str) and t.strip():
            return canon_name(t)
    return None

def now_iso():
    # timezone-aware ISO → Z
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

def log_event(user_msg, term, ans, candidates=None):
    rec = {
        "ts": now_iso(),
        "user": user_msg,
        "term": term if isinstance(term, str) else None,
        "candidates": candidates if isinstance(term, list) else None,
        "answer_found": bool(ans),
        "answer": ans if ans else None
    }
    history.append(rec)
    if save_enabled:
        with open(HISTORY_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def print_history(n=20):
    items = history[-n:] if n>0 else history
    if not items:
        print("Henüz geçmiş yok.\n")
        return
    print(f"\nSon {len(items)} kayıt:")
    for i, rec in enumerate(items, 1):
        tag = "✅" if rec.get("answer_found") else ("🤔" if rec.get("candidates") else "❌")
        if rec.get("candidates"):
            cand = ", ".join(rec["candidates"])
            print(f"{i:02d}. [{rec['ts']}] {tag} Soru: {rec['user']}  → Adaylar: {cand}")
        else:
            print(f"{i:02d}. [{rec['ts']}] {tag} Soru: {rec['user']}")
            if rec.get("term"):
                print(f"    Terim: {rec['term']}")
            if rec.get("answer"):
                print(f"    Cevap: {rec['answer']}")
    print("")

def export_markdown(path=EXPORT_MD_PATH):
    if not history:
        print("ℹ Dışa aktarılacak kayıt yok.")
        return
    lines = ["# Chat Geçmişi\n"]
    for rec in history:
        tag = "✅" if rec.get("answer_found") else ("🤔" if rec.get("candidates") else "❌")
        lines.append(f"## {tag} {rec['ts']}")
        lines.append(f"**Soru:** {rec['user']}")
        if rec.get("candidates"):
            lines.append(f"**Aday Terimler:** {', '.join(rec['candidates'])}")
        if rec.get("term"):
            lines.append(f"**Terim:** {rec['term']}")
        if rec.get("answer"):
            lines.append(f"**Cevap:** {rec['answer']}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding='utf-8')
    print(f" Markdown dışa aktarıldı: {path}")

def toggle_save(on: bool):
    global save_enabled
    save_enabled = on
    state = "açık" if on else "kapalı"
    print(f" Kayıt modu {state}.")

def clear_history():
    global history
    history = []
    try:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
        if os.path.exists(EXPORT_MD_PATH):
            os.remove(EXPORT_MD_PATH)
    except Exception as e:
        print(f" Dosya silinirken hata: {e}")
    print("🧹 Geçmiş sıfırlandı.")

HELP_TEXT = """
🔧 Komutlar:
  /gecmis               → son 20 kaydı göster
  /gecmis 50            → son 50 kaydı göster
  /exportmd             → geçmişi Markdown olarak kaydet
  /kaydet ac|kapat      → dosyaya loglamayı aç/kapat
  /silgecmis            → geçmişi ve dosyayı sil
  /ogret TERIM: TANIM   → yeni bir terim ve tanımı öğret (kalıcı kaydetmeye çalışır)
  /ogren TERIM          → etkileşimli olarak TERIM'i öğren (hızlı)
  /ogren TERIM: TANIM   → doğrudan TERIM'i kaydet
  /yardim               → bu menü
  (komut değilse normal soru olarak işlenir)
"""

# === Generatif Model (yükleme + retrieval + üretim) ===
BASE_DIR = Path(__file__).resolve().parent

def _resolve_ckpt():
    # 1) DOMAIN_LLM_CKPT env
    env_p = os.environ.get('DOMAIN_LLM_CKPT', '').strip()
    if env_p and os.path.exists(env_p):
        return env_p
    # 2) Local next to this script
    local = BASE_DIR / 'gpt_gen_checkpoint.pth'
    if local.exists():
        return str(local)
    # 3) CWD
    cwd = Path.cwd() / 'gpt_gen_checkpoint.pth'
    if cwd.exists():
        return str(cwd)
    # 4) Colab default
    colab = Path('/content/gpt_gen_checkpoint.pth')
    if colab.exists():
        return str(colab)
    return ''

CKPT_PATH = _resolve_ckpt()
# Opsiyonel genel model (alan dışı sorular için)
CKPT_GENERAL = os.environ.get('GENERAL_LLM_CKPT', '').strip()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
model_general = None
itos = {}
stoi = {}
encode = None
decode = None
pad_token = '<pad>'
sep_token = '</s>'
block_size = 256

def _load_ckpt(path):
    global itos, stoi, pad_token, sep_token, block_size
    if not path or not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    itos = ckpt['itos']; stoi = ckpt['stoi']
    cfg = ckpt.get('config', {})
    block_size = cfg.get('block_size', 256)
    n_embd  = cfg.get('n_embd', 512)
    n_layer = cfg.get('n_layer', 8)
    n_head  = cfg.get('n_head', 8)
    dropout = cfg.get('dropout', 0.1)
    pad_token = cfg.get('pad_token', pad_token)
    sep_token = cfg.get('sep_token', sep_token)

    model_local = GPTLanguageModel(
        vocab_size=len(itos),
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout
    ).to(device)
    model_local.load_state_dict(ckpt['model_state_dict'])
    model_local.eval()
    return model_local

def load_generative():
    global model, model_general, itos, stoi, encode, decode
    model = _load_ckpt(CKPT_PATH)
    # Genel model ayrı checkpoint ise yükle
    if CKPT_GENERAL and os.path.exists(CKPT_GENERAL):
        model_general = _load_ckpt(CKPT_GENERAL)
    else:
        model_general = None
    # Only set enc/dec if a model is loaded and vocab exists
    if model is not None and isinstance(stoi, dict) and stoi:
        def _enc_safe(s):
            any_idx = next(iter(stoi.values()))
            return [stoi.get(c, any_idx) for c in s]
        def _dec_safe(l):
            return ''.join(itos.get(i, '') for i in l)
        encode = _enc_safe
        decode = _dec_safe
    else:
        encode = None
        decode = None
    return model is not None

_ = load_generative()

def serialize_record(name, rec):
    parts = [f"{canon_name(name)}:"]
    for k, v in rec.items():
        parts.append(f"{k}: {v}")
    return " ".join(parts)

def top_k_records(query, k=8):
    nq = norm(query)
    hits = set()
    # 0) Sorguda açık terimler varsa öncelik ver
    terms_in_q = extract_terms_raw(query)
    if terms_in_q:
        for t in terms_in_q:
            hits.add(canon_name(t))
        return list(sorted(hits))[:k]
    # 1) Özellik/kategori eşleşmesi (alias’larla)
    cat_root = extract_feature_from_query(query)
    if cat_root:
        keys = [norm(cat_root)] + [norm(s) for s in FEATURE_ALIASES.get(cat_root, [])]
        for k in keys:
            hits |= set(feature2terms.get(k, set()))
    # 2) kaba anahtar kelime eşleşmesi
    toks = [t for t in re.split(r'[\s,;:]+', nq) if t]
    for name, rec in props.items():
        hay = norm(name) + ' ' + norm(' '.join(rec.keys())) + ' ' + norm(' '.join(rec.values()))
        if any(t in hay for t in toks):
            hits.add(canon_name(name))
    return list(sorted(hits))[:k]

def build_context(cands):
    lines = []
    for name, rec in props.items():
        base = canon_name(name)
        if base in cands:
            lines.append(serialize_record(name, rec))
    return " | ".join(lines)

@torch.no_grad()
def generate_text(prompt, max_new_tokens=200, temperature=0.9, top_k=0):
    if (model is None) or (encode is None) or (decode is None):
        return None
    start = "<s>" + prompt
    idx = torch.tensor([encode(start)], dtype=torch.long).to(device)
    generated_tail = ""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = torch.softmax(logits / max(1e-5, temperature), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
        # incremental decode of just the last char
        ch = globals().get('itos', {}).get(int(next_token.item()), '')
        generated_tail += ch
        # Erken durdurma: ayraç üretildiğinde kes
        if sep_token and sep_token in generated_tail:
            break
    full = decode(idx[0].tolist())
    return full[len("<s>"):]

def _extract_answer_only(generated_text: str) -> str:
    if not generated_text:
        return ""
    # Son 'Cevap:' işaretinden sonrasını al
    lower = generated_text.lower()
    key = 'cevap:'
    pos = lower.rfind(key)
    ans = generated_text[pos+len(key):] if pos != -1 else generated_text
    # Bir sonraki ayraç gelirse oraya kadar kırp
    cut = ans.split(sep_token)[0] if sep_token in ans else ans
    return cut.strip()

def answer_generative(query):
    # --- Erken dönüşler: alan sorularını ve özellik listesini generatiften ÖNCE yanıtla ---
    intent, fld = detect_intent(query)
    term_one = extract_term(query)

    # ÖZELLİK LİSTESİ (örn: "atmaca özellikleri")
    if intent == 'features' and term_one:
        feats_off, feats_user = get_features_list(term_one)
        if feats_off:
            return "Özellikler: " + ", ".join(feats_off)
        if feats_user:
            return "Özellikler (kullanıcıdan): " + ", ".join(feats_user)
        # yoksa fallback'e izin ver
        return ""

    # BELİRLİ ALAN (örn: "atmaca türü", "SOM menşei")
    if intent == 'field' and fld and term_one:
        val, src, conflict = get_field_value(term_one, fld)
        if val:
            prefix = "(Resmi) " if src == 'official' else "(Kullanıcı) "
            tail = f" {conflict}" if conflict else ""
            return f"{prefix}{fld}: {val}{tail}"
        return ""  # fallback'e izin ver

    # TANIM (varsa doğrudan dön)
    if intent == 'definition' and term_one:
        key_try1 = term_one; key_try2 = canon_name(term_one)
        if key_try1 in d: return d[key_try1]
        if key_try2 in d: return d[key_try2]
        ename, ptxt = props_for_entity(term_one)
        if ptxt: return ptxt

    # KATEGORİ / LİSTELEME
    if intent == 'feature_list':
        items = list_by_feature(query)
        if wants_count(query):
            return (f" Toplam: {len(items)}\n" + ("\n".join(f"- {n}" for n in items) if items else ""))
    # Genel liste
    if intent == 'list':
        items = list_by_keyword(query)
        if wants_count(query):
            return (f" Toplam: {len(items)}\n" + ("\n".join(f"- {n}" for n in items) if items else ""))

    # --- Generatif üretim (retrieval ile) ---
    if model is None:
        return None
    cands = top_k_records(query, k=8)
    ctx = build_context(cands)
    prompt = (f"Soru: {query} {sep_token} Bağlam: {ctx} {sep_token} Cevap: "
              f"Lütfen mümkünse tek satırda özetle. Gerekirse şu biçimi kullan: "
              f"Tanım: ... | Tür: ... | Menşei: ... | Özellikler: a, b, c")
    raw = generate_text(prompt, max_new_tokens=220, temperature=0.85, top_k=50)
    ans_only = _extract_answer_only(raw)

    # Güçlü fallback
    if len(ans_only) < 5:
        if term_one and term_one in d:
            return d[term_one]
        ename, ptxt = props_for_entity(term_one) if term_one else (None, None)
        if ptxt:
            return ptxt
        lst = list_by_feature(query)
        if lst:
            if wants_count(query):
                return f"🔢 Toplam: {len(lst)}\n" + "\n".join(f"- {n}" for n in lst)
            return "\n".join(f"- {n}" for n in lst)
    return ans_only or raw

def answer_general(query):
    # Alan dışı sorular için genel model; bağlam vermeden direkt üretim
    if model_general is None:
        return None
    start = f"Soru: {query} {sep_token} Cevap: "
    raw = generate_text(start, max_new_tokens=200, temperature=0.9, top_k=50)
    return _extract_answer_only(raw)

# --- CLI döngüsü ---
print("🛰️ Sözlük + Özellik arayüzü yüklendi. Çıkmak için Ctrl+C / Ctrl+D.")
print(HELP_TEXT)
print("🤖 Generatif model: " + ("yüklendi." if model is not None else "bulunamadı, kural tabanlı mod kullanılacak."))

while True:
    try:
        q = input(" Prompt Girin: ")
    except (EOFError, KeyboardInterrupt):
        print("\n Görüşmek üzere!")
        break

    if not q.strip():
        continue

    if q.startswith('/'):
        parts = q.strip().split()
        cmd = parts[0].lower()
        if cmd in ('/yardim', '/help'):
            print(HELP_TEXT); continue
        if cmd in ('/gecmis', '/geçmiş'):
            n = 20
            if len(parts) > 1 and parts[1].isdigit():
                n = int(parts[1])
            print_history(n); continue
        if cmd == '/exportmd':
            export_markdown(); continue
        if cmd == '/kaydet':
            if len(parts) > 1 and parts[1].lower() in ('ac','aç','on'):
                toggle_save(True)
            elif len(parts) > 1 and parts[1].lower() in ('kapat','off'):
                toggle_save(False)
            else:
                print("Kullanım: /kaydet ac | /kaydet kapat")
            continue
        if cmd == '/ogret':
            # kalan tüm mesajı al ve öğret
            payload = q[len(parts[0]):].strip()
            ok, msg = teach_definition_inline(payload)
            print(msg)
            continue
        if cmd == '/ogren':
            payload = q[len(parts[0]):].strip()
            # Doğrudan "TERIM: TANIM" verildiyse /ogret gibi işle
            if ':' in payload:
                ok, msg = teach_definition_inline(payload)
                print(msg); continue
            # Sadece terim verildiyse hızlı etkileşimli giriş
            term = payload.strip()
            if not term:
                print("Kullanım: /ogren TERIM  veya  /ogren TERIM: TANIM"); continue
            print(f"↳ {term} için kısa tanım girin (boş bırakılırsa iptal):")
            try:
                tan = input("> Tanım: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("İptal."); continue
            if not tan:
                print("İptal."); continue
            ok, msg = teach_definition_inline(f"{term}: {tan}")
            print(msg); continue
        if cmd == '/silgecmis':
            clear_history(); continue
        print(" Bilinmeyen komut. /yardim yazabilirsin.")
        continue

    # Çoklu terim tespiti: tanım/özellik/alan niyetlerinde önce kullanıcıdan seçim iste
    intent, _fld = detect_intent(q)
    terms_in_q = extract_terms_raw(q)

    # --- Coreference patch: tekil terim bulunduysa geçerli konuyu güncelle ---
    if len(terms_in_q) == 1:
        set_current_subject(terms_in_q[0])

    if len(terms_in_q) > 1 and intent in ('definition','features','field'):
        print(" Birden fazla aday buldum, hangisini kastettiniz?\n- " + "\n- ".join(terms_in_q) + "\n")
        log_event(q, terms_in_q, None, candidates=terms_in_q)
        continue

    # Bilinmeyen terim/alanları aktif öğren: eğer resmi+öğrenilmişte yoksa kullanıcıdan iste
    # (Yalnızca tanım/özellik/tür/menşei niyetlerinde)
    intent, field_name = detect_intent(q)
    terms_in_q = extract_terms_raw(q)
    if len(terms_in_q) == 1 and intent in ('definition','features','field'):
        t = terms_in_q[0]
        if intent == 'definition':
            has_def = (t in d) or ('Tanım' in learned_props.get(t, {}))
            if not has_def:
                print(f" {t} için tanım kayıtlı değil. Kısa tanımını, türünü, menşeini ve 2-3 önemli özelliğini yazar mısın?")
        elif intent == 'features':
            off, usr = get_features_list(t)
            if not off and not usr:
                print(f"{t} için özellik bulunmadı. 2-3 temel özelliğini yazar mısın?")
        elif intent == 'field' and field_name in ('Tür','Menşei'):
            val, src, _ = get_field_value(t, field_name)
            if not val:
                print(f"{t} için {field_name} bilgisi yok. {field_name} nedir?")

    # Önce generatif cevap dene (artık alan/özellik sorularında erken dönüş veriyor)
    gen = answer_generative(q)
    if gen and gen.strip():
        # --- Coreference patch: generatif cevapta da tekil terimi güncelle ---
        one = extract_term(q)
        if one:
            set_current_subject(one)
        print(f"\n CEVAP (LLM):\n{gen}\n")
        log_event(q, None, gen)
        continue
    # Alan dışı ise genel model dene
    gen2 = answer_general(q)
    if gen2 and gen2.strip():
        one = extract_term(q)
        if one:
            set_current_subject(one)
        print(f"\n CEVAP (GENEL LLM):\n{gen2}\n")
        log_event(q, None, gen2)
        continue

    # Generatif yoksa kural tabanlıya düş
    def answer_for(q: str):
        last_subject = None  # local use
        intent, field = detect_intent(q)
        terms_in_q = extract_terms_raw(q)  # çoklu aday kontrolü
        term = terms_in_q[0] if len(terms_in_q) == 1 else None
        want_cnt = wants_count(q)

        # Çoklu aday
        if len(terms_in_q) > 1 and intent in ('definition','features','field'):
            return terms_in_q, None

        # sınıflandırma
        if intent == 'classify':
            include_lists = any(h in norm(q) for h in LIST_HINTS)
            text = " Özelliklere göre dağılım:\n" + summarize_feature_counts(include_lists=include_lists or want_cnt)
            return None, text

        if intent == 'type_group':
            txt = summarize_types(feature_filter_root=None)
            return None, txt

        if intent == 'feature_list':
            found = list_by_feature(q)
            if want_cnt:
                return None, (f" Toplam: {len(found)}\n" + ("\n".join(f"- {n}" for n in found) if found else ""))
            if found:
                return None, "\n".join(f"- {n}" for n in found)
            else:
                return None, " Bu kategoriyle eşleşen kayıt bulunamadı."

        if intent == 'list':
            found = list_by_keyword(q)
            if want_cnt:
                return None, (f" Toplam: {len(found)}\n" + ("\n".join(f"- {n}" for n in found) if found else ""))
            if found:
                return None, "\n".join(f"- {n}" for n in found)
            else:
                return None, " Kriterle eşleşen bulunamadı."

        if intent == 'definition':
            if term:
                key_try1 = term
                key_try2 = canon_name(term) if term else None
                ans = d.get(key_try1) or (d.get(key_try2) if key_try2 else None)
                if ans:
                    return key_try1 if key_try1 in d else key_try2, ans
                lp = learned_props.get(term, {})
                if 'Tanım' in lp:
                    return term, f"(Kullanıcı) Tanım: {lp['Tanım']}"
            cat = extract_feature_from_query(q)
            if cat and cat in FEATURE_DEFINITIONS and not term:
                return cat.title(), FEATURE_DEFINITIONS[cat]
            if STRICT_DEFINITION:
                if term:
                    ask = f"{term}’in kısa tanımı, türü, menşei ve 2-3 önemli özelliği nedir?"
                else:
                    ask = "Bahsettiğiniz terimin kısa tanımı, türü, menşei ve 2-3 önemli özelliği nedir?"
                return term, f" Tanım kaydı bulunamadı. {ask} (Cevabınızı öğrendikten sonra kullanacağım.)"
            ename, ptxt = props_for_entity(term) if term else (None, None)
            if ename and ptxt:
                return ename, f"Tanım bulunamadı; kısa özet: {ptxt}"

        if intent == 'features':
            if term is None:
                return None, " Hangi varlığın özellikleri? (Örn: 'UMTAS özellikleri')"
            feats_off, feats_user = get_features_list(term)
            if feats_off:
                note = ""
                if feats_user:
                    extra = [f for f in feats_user if f not in feats_off]
                    if extra:
                        note = f"\n(Not: Oturumda eklenenler: {', '.join(extra)})"
                return term, ("Özellikler:\n" + "\n".join(f"- {f}" for f in feats_off) + note)
            if feats_user:
                return term, ("Özellikler (kullanıcıdan):\n" + "\n".join(f"- {f}" for f in feats_user))
            return None, " Özellik bilgisi bulunamadı. '/ogret TERIM: Özellik1, Özellik2, ...' ile ekleyebilirsin."

        if intent == 'field' and field:
            if term is None:
                return None, f" Hangi varlık için '{field}'?"
            val, src, conflict = get_field_value(term, field)
            if val:
                prefix = "(Resmi) " if src == 'official' else "(Kullanıcı) "
                tail = f" {conflict}" if conflict else ""
                return term, f"{prefix}{field}: {val}{tail}"

        if term and term in d and not (intent in ('features','field','definition')):
            ans = d.get(term)
            if ans:
                return term, ans

        ename, ptxt = props_for_entity(term) if term else (None, None)
        if ptxt:
            return ename, ptxt

        return term, d.get(term)

    term, ans = answer_for(q)

    if term is None and ans is None:
        print(" Tanım/özellik bulunamadı.\n")
        log_event(q, term, None)
        continue

    if isinstance(term, list):
        print(" Birden fazla aday buldum, hangisini kastettiniz?\n- " + "\n- ".join(term) + "\n")
        log_event(q, term, None, candidates=term)
        continue

    if ans:
        # --- Coreference patch: başarıyla cevaplanan tekil terimi geçerli konu yap ---
        if isinstance(term, str) and term.strip():
            set_current_subject(term)
        print(f"\n CEVAP: {term if term else '(liste)'}\n{ans}\n")
        log_event(q, term, ans)
    else:
        print("Tanım/özellik bulunamadı.\n")
        log_event(q, term, None)

