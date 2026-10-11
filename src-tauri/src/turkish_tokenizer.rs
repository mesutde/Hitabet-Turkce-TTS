use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// Gercek model alfabesi (models/onnx/vocab.json, 49 token):
// 0=<pad>, 1=<unk>, 2=' ', noktalama ve a-z + c ö ü ğ ı ş
pub const PAD_TOKEN: i64 = 0;
pub const UNK_TOKEN: i64 = 1;

pub struct TurkishTokenizer {
    stoi: HashMap<char, i64>,
    vocab_set: HashSet<char>,
    unk_id: i64,
}

impl TurkishTokenizer {
    pub fn from_vocab_file(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref()).with_context(|| {
            format!("vocab.json okunamadi: {}", path.as_ref().display())
        })?;
        let vocab: Vec<String> =
            serde_json::from_str(&data).context("vocab.json ayrıştırılamadı")?;
        let mut stoi = HashMap::new();
        let mut vocab_set = HashSet::new();
        for (i, tok) in vocab.iter().enumerate() {
            if tok.chars().count() == 1 {
                let ch = tok.chars().next().unwrap();
                stoi.insert(ch, i as i64);
                vocab_set.insert(ch);
            }
        }
        let unk_id = vocab
            .iter()
            .position(|t| t == "<unk>")
            .map(|i| i as i64)
            .unwrap_or(UNK_TOKEN);
        Ok(Self {
            stoi,
            vocab_set,
            unk_id,
        })
    }

    /// Konusulur metin (normalize + alfabe, bitis duzeltmesiz).
    pub fn preprocess(&self, text: &str) -> String {
        let normalized = normalize_turkish_text(text);
        apply_alphabet(&normalized, &self.vocab_set)
    }

    /// Konusulur metin parcasi -> harf id listesi. Gercek modelde BOS/EOS/pad
    /// YOK; mask = ids != 0. Bilinmeyen harf -> <unk> (1).
    pub fn encode_ids(&self, spoken: &str) -> Vec<i64> {
        spoken
            .chars()
            .map(|ch| self.stoi.get(&ch).copied().unwrap_or(self.unk_id))
            .collect()
    }

    pub fn vocab_chars(&self) -> &HashSet<char> {
        &self.vocab_set
    }
}

/// chunker.py aynasi: konusulur metni parcalara boler.
/// ~10 sn'yi asan metin son iyi noktadan kesilir: cumle sonu, sonra
/// virgul/noktali/iki nokta, sonra bosluk; hicbiri yoksa sert kesim.
/// Donus: (parca, sonrasindaki sessizlik sn). Son parcanin duraklamasi 0'dir.
pub const LETTERS_PER_SECOND: f32 = 18.0;
pub const MAX_SECONDS: f32 = 10.0;
pub const MAX_LETTERS: usize = 250;
pub const SENTENCE_PAUSE: f32 = 0.25;
pub const CLAUSE_PAUSE: f32 = 0.12;

pub fn chunk_spoken(text: &str, speed: f32) -> Vec<(String, f32)> {
    let limit = (MAX_LETTERS as f32).min(LETTERS_PER_SECOND * MAX_SECONDS * speed.max(0.25)) as usize;
    let mut pieces: Vec<(String, f32)> = Vec::new();
    let mut rest = text.trim().to_string();
    while !rest.is_empty() {
        let rest_len = rest.chars().count();
        let (cut, pause) = if rest_len > limit {
            find_cut(&rest, limit)
        } else {
            (rest_len, 0.0)
        };
        let cut_byte = char_to_byte_idx(&rest, cut);
        let piece = rest[..cut_byte].trim().to_string();
        rest = rest[cut_byte..].trim().to_string();
        if piece.chars().any(|c| c.is_alphabetic()) {
            pieces.push((finish_piece(&piece), pause));
        }
    }
    if let Some(last) = pieces.last_mut() {
        last.1 = 0.0;
    }
    pieces
}

fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .map(|(b, _)| b)
        .nth(char_idx)
        .unwrap_or(s.len())
}

/// Verilen sinir icinde son iyi kesim noktasi (harf sayisi, duraklama sn).
fn find_cut(rest: &str, limit: usize) -> (usize, f32) {
    let chars: Vec<char> = rest.chars().collect();
    let n = (limit + 1).min(chars.len());
    // 1) cumle sonu: [.!?]+["']*(?= )
    let mut at = 0;
    let mut sentence_end: Option<usize> = None;
    while at < n {
        if ".!?".contains(chars[at]) {
            let mut j = at + 1;
            while j < n && ".!?".contains(chars[j]) {
                j += 1;
            }
            while j < n && (chars[j] == '"' || chars[j] == '\'') {
                j += 1;
            }
            if j < n && chars[j] == ' ' {
                sentence_end = Some(j);
            }
            at = j.max(at + 1);
        } else {
            at += 1;
        }
    }
    if let Some(cut) = sentence_end {
        return (cut, SENTENCE_PAUSE);
    }
    // 2) cumlecik: [,;:](?= )
    let mut clause_end: Option<usize> = None;
    for i in 0..n {
        if ",;:".contains(chars[i]) && i + 1 < n && chars[i + 1] == ' ' {
            clause_end = Some(i + 1);
        }
    }
    if let Some(cut) = clause_end {
        return (cut, CLAUSE_PAUSE);
    }
    // 3) bosluk oncesi: \S(?= )
    let mut space_end: Option<usize> = None;
    for i in 0..n {
        if !chars[i].is_whitespace() && i + 1 < n && chars[i + 1] == ' ' {
            space_end = Some(i + 1);
        }
    }
    if let Some(cut) = space_end {
        return (cut, CLAUSE_PAUSE);
    }
    // 4) sert kesim (duraklamasiz).
    (limit.min(chars.len()), 0.0)
}

/// Adim 1: konusma-dili normalizasyonu (normalizer-tr yaklasigi).
/// Rakam/tarih/saat/para birim acilimi + kisaltma genisletme.
pub fn normalize_turkish_text(text: &str) -> String {
    let lower = text.trim().to_string();
    let words: Vec<String> = lower
        .split_whitespace()
        .map(|w| expand_abbreviation(w))
        .collect();
    expand_numbers(&words.join(" "))
}

fn expand_abbreviation(word: &str) -> String {
    let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
    // Kucuk harfe henuz cevrilmedi; karsilastirma icin Turkce lower uygula
    let key = tr_lowercase_char_str(&clean);

    let abbreviations: &[(&str, &str)] = &[
        ("dr", "doktor"),
        ("prof", "profesör"),
        ("doç", "doçent"),
        ("sn", "sayın"),
        ("vs", "ve saire"),
        ("vb", "ve benzeri"),
        ("vd", "ve devamı"),
        ("ltd", "limited"),
        ("şti", "şirketi"),
        ("aş", "anonim şirketi"),
        ("kg", "kilogram"),
        ("km", "kilometre"),
        ("cm", "santimetre"),
        ("mm", "milimetre"),
        ("ml", "mililitre"),
        ("gr", "gram"),
        ("dk", "dakika"),
    ];

    for (abbr, full) in abbreviations {
        if key == *abbr {
            return full.to_string();
        }
    }
    word.to_string()
}

fn expand_numbers(text: &str) -> String {
    let mut result = String::new();
    let mut num_buf = String::new();

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            num_buf.push(ch);
        } else {
            if !num_buf.is_empty() {
                if let Ok(n) = num_buf.parse::<i64>() {
                    result.push_str(&number_to_turkish(n));
                } else {
                    result.push_str(&num_buf);
                }
                num_buf.clear();
            }
            result.push(ch);
        }
    }

    if !num_buf.is_empty() {
        if let Ok(n) = num_buf.parse::<i64>() {
            result.push_str(&number_to_turkish(n));
        } else {
            result.push_str(&num_buf);
        }
    }

    result
}

fn number_to_turkish(n: i64) -> String {
    if n == 0 {
        return "sıfır".to_string();
    }

    let ones = [
        "", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
    ];
    let tens = [
        "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
    ];

    if n < 0 {
        return format!("eksi {}", number_to_turkish(-n));
    }
    if n < 10 {
        return ones[n as usize].to_string();
    }
    if n < 100 {
        let t = tens[(n / 10) as usize];
        let o = ones[(n % 10) as usize];
        if o.is_empty() {
            return t.to_string();
        }
        return format!("{} {}", t, o);
    }
    if n < 1000 {
        let h = format!("{} yüz", ones[(n / 100) as usize]);
        let rem = n % 100;
        if rem == 0 {
            return h;
        }
        return format!("{} {}", h, number_to_turkish(rem));
    }
    if n < 1_000_000 {
        let th = format!("{} bin", number_to_turkish(n / 1000));
        let rem = n % 1000;
        if rem == 0 {
            return th;
        }
        return format!("{} {}", th, number_to_turkish(rem));
    }
    if n < 1_000_000_000 {
        let m = format!("{} milyon", number_to_turkish(n / 1_000_000));
        let rem = n % 1_000_000;
        if rem == 0 {
            return m;
        }
        return format!("{} {}", m, number_to_turkish(rem));
    }

    let b = format!("{} milyar", number_to_turkish(n / 1_000_000_000));
    let rem = n % 1_000_000_000;
    if rem == 0 {
        return b;
    }
    format!("{} {}", b, number_to_turkish(rem))
}

/// Adim 2: frontend.py::alphabet ile ayni is:
/// tipografi duzeltme -> Turkce lower (İ->i, I->ı) -> vocab disi karakteri bosluga cevir.
pub fn apply_alphabet(text: &str, vocab: &HashSet<char>) -> String {
    const TURKISH_SPECIAL: [char; 12] = [
        'ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü',
    ];
    let is_tr = |c: char| TURKISH_SPECIAL.contains(&c);

    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let ch = match ch {
            '’' | '‘' | 'ʼ' | '´' | '`' => '\'',
            '“' | '”' | '„' | '«' | '»' => '"',
            '–' | '—' | '−' => '-',
            '…' => {
                out.push_str("...");
                continue;
            }
            c => c,
        };
        // Turkce lower: once İ->i, I->ı (Rust lower() İ->i̇ yapar, yanlis olur)
        let lowered: String = if ch == 'İ' {
            "i".to_string()
        } else if ch == 'I' {
            "ı".to_string()
        } else {
            ch.to_lowercase().collect()
        };
        for mut c in lowered.chars() {
            if !is_tr(c) {
                c = strip_combining(c);
            }
            if c.len_utf8() == 0 {
                continue;
            }
            // NFKD sonrasi birden cok karakter cikabilir (örn. 'æ'->'ae'); tek tek ele
            for sc in c.to_string().chars() {
                if vocab.contains(&sc) {
                    out.push(sc);
                } else if sc.is_whitespace() {
                    out.push(' ');
                } else {
                    out.push(' ');
                }
            }
        }
    }
    // bosluklari tekle
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_space = true; // bastaki boslugu at
    for ch in out.chars() {
        if ch == ' ' {
            if !prev_space {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }
    collapsed.trim_end().to_string()
}

/// Parca bitirici (chunker.py::finish ile birebir): model, cumle sonuyla
/// bitecek sekilde egitildi. Sonu nokta/unlem/soru isaretiyle bitmeyen
/// metin, havada kesilmis gibi biter (son ornekler sifira inmez -> tik sesi).
/// Bu yuzden eksikse "." eklenir; web surumunde bu hic yasanmaz.
pub fn finish_piece(text: &str) -> String {
    let t = text.trim_end();
    if t.is_empty() {
        return String::new();
    }
    let stripped: String = t.trim_end_matches(['"', '\'', ')']).to_string();
    match stripped.chars().last() {
        Some('.') | Some('!') | Some('?') => t.to_string(),
        _ => {
            let base = t.trim_end_matches([',', ';', ':', '-', ' ']);
            if base.is_empty() {
                String::new()
            } else {
                format!("{base}.")
            }
        }
    }
}

/// Birlesik aksanli Latin harfini tabana indir (NFKD + combining strip yaklasigi).
fn strip_combining(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' => 'i',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ý' | 'ÿ' => 'y',
        'ñ' => 'n',
        'ß' => 's',
        'æ' => 'a', // tek char donmeliyiz; 'ae' yerine 'a'
        'œ' => 'o',
        'ç' => 'c',
        'ğ' => 'g',
        'ş' => 's',
        _ => c,
    }
}

fn tr_lowercase_char_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch == 'İ' {
            out.push('i');
        } else if ch == 'I' {
            out.push('ı');
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turkish_lowercase() {
        let vocab: HashSet<char> = "abcçdefgğhıijklmnoöprsştuüvyz !".chars().collect();
        assert_eq!(apply_alphabet("İI", &vocab), "iı");
    }

    #[test]
    fn test_expand_digits() {
        assert_eq!(normalize_turkish_text("3 elma"), "3 elma".replace("3", "üç"));
    }

    #[test]
    fn test_number_to_turkish_large() {
        assert_eq!(number_to_turkish(2024), "iki bin yirmi dört");
    }

    #[test]
    fn test_chunk_short_text_single_piece() {
        assert_eq!(
            chunk_spoken("merhaba dünya.", 1.0),
            vec![("merhaba dünya.".to_string(), 0.0)]
        );
        // Bitis isaretsiz metin cumle sonuyla bitirilir.
        assert_eq!(
            chunk_spoken("merhaba dünya", 1.0),
            vec![("merhaba dünya.".to_string(), 0.0)]
        );
    }

    #[test]
    fn test_chunk_cuts_at_sentence_end() {
        let text = format!("cümle bir. {}", "kelime ".repeat(60));
        let pieces = chunk_spoken(&text, 1.0);
        assert!(pieces.len() > 1);
        assert_eq!(pieces[0], ("cümle bir.".to_string(), SENTENCE_PAUSE));
        assert_eq!(pieces.last().unwrap().1, 0.0);
        for (p, _) in &pieces {
            assert!(p.chars().count() <= MAX_LETTERS);
        }
    }

    #[test]
    fn test_chunk_cuts_at_clause() {
        let text = format!("ahmet, {}", "kelime ".repeat(60));
        let pieces = chunk_spoken(&text, 1.0);
        assert!(pieces.len() > 1);
        assert_eq!(pieces[0].0, "ahmet.");
        assert!((pieces[0].1 - CLAUSE_PAUSE).abs() < 1e-6);
    }

    #[test]
    fn test_finish_piece() {
        assert_eq!(finish_piece("merhaba"), "merhaba.");
        assert_eq!(finish_piece("merhaba."), "merhaba.");
        assert_eq!(finish_piece("merhaba!"), "merhaba!");
        assert_eq!(finish_piece("merhaba?"), "merhaba?");
        assert_eq!(finish_piece("dedi.\""), "dedi.\"");
        assert_eq!(finish_piece("hızlı,"), "hızlı.");
        assert_eq!(finish_piece("  "), "");
    }
}
