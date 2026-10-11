use anyhow::{Context, Result};
use fancy_regex::{Captures, Regex};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

// Gercek model alfabesi (models/antalia/symbols.json, 47 token):
// 0=<pad>, 1=<bos>, 2=<eos>, noktalama " .,?!'-:;", a-z + c g i o s u + a i u
pub const PAD_TOKEN: i64 = 0;
#[allow(dead_code)]
pub const BOS_TOKEN: i64 = 1;
#[allow(dead_code)]
pub const EOS_TOKEN: i64 = 2;

pub struct AntaliaTokenizer {
    stoi: HashMap<char, i64>,
    vocab_set: HashSet<char>,
    bos_id: i64,
    eos_id: i64,
}

impl AntaliaTokenizer {
    pub fn from_symbols_file(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref()).with_context(|| {
            format!("symbols.json okunamadi: {}", path.as_ref().display())
        })?;
        let symbols: Vec<String> =
            serde_json::from_str(&data).context("symbols.json ayrıştırılamadı")?;
        let mut stoi = HashMap::new();
        let mut vocab_set = HashSet::new();
        for tok in symbols.iter() {
            if tok.chars().count() == 1 {
                let ch = tok.chars().next().unwrap();
                if let Some(pos) = symbols.iter().position(|t| t == tok) {
                    stoi.insert(ch, pos as i64);
                    vocab_set.insert(ch);
                }
            }
        }
        let id_of = |name: &str| {
            symbols
                .iter()
                .position(|t| t == name)
                .map(|i| i as i64)
                .unwrap_or(1)
        };
        Ok(Self {
            stoi,
            vocab_set,
            bos_id: id_of("<bos>"),
            eos_id: id_of("<eos>"),
        })
    }

    /// Ham metin -> konusulur metin (normalize + alfabe).
    pub fn preprocess(&self, text: &str) -> String {
        let spoken = normalize_for_model_text(text);
        apply_alphabet(&spoken, &self.vocab_set)
    }

    /// Konusulur metin -> id listesi ([BOS] ... [EOS], bilinmeyen harf ATLANIR).
    pub fn encode_ids(&self, spoken: &str) -> Vec<i64> {
        let mut ids = vec![self.bos_id];
        for ch in spoken.chars() {
            if let Some(&id) = self.stoi.get(&ch) {
                ids.push(id);
            }
        }
        ids.push(self.eos_id);
        ids
    }

    pub fn vocab_chars(&self) -> &HashSet<char> {
        &self.vocab_set
    }

    /// Bosluk harfi id'si (kelime sinirlari icin; vocab'dan okunur).
    pub fn space_id(&self) -> i64 {
        self.stoi.get(&' ').copied().unwrap_or(3)
    }
}

// ---------------------------------------------------------------------------
// Alfabe temizligi (clean_model_text aynasi)
// ---------------------------------------------------------------------------

pub fn apply_alphabet(text: &str, vocab: &HashSet<char>) -> String {
    let folded: String = text
        .chars()
        .map(|c| match c {
            '\u{2019}' | '\u{2018}' | '`' | '\u{00b4}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{00ab}' | '\u{00bb}' => ' ',
            '\u{2013}' | '\u{2014}' => '-',
            '\u{2026}' => '.',
            _ => c,
        })
        .collect();
    let mut out = String::with_capacity(folded.len());
    for ch in folded.chars() {
        if vocab.contains(&ch) || ch.is_whitespace() {
            out.push(ch);
        } else {
            out.push(' ');
        }
    }
    // bosluk-noktalama duzeltme: \s+([.,?!:;]) -> \1 ; bosluklari tekle
    let out = SP_PUNCT_RE
        .get_or_init(|| re(r"\s+([.,?!:;])"))
        .replace_all(&out, "$1")
        .into_owned();
    collapse_spaces(&out)
}

fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = true;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim_end().to_string()
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("gecersiz regex")
}

fn rei(pattern: &str) -> Regex {
    Regex::new(&format!("(?i){pattern}")).expect("gecersiz regex")
}

static SP_PUNCT_RE: OnceLock<Regex> = OnceLock::new();

// ---------------------------------------------------------------------------
// Turkce kucultme + sayi sozleri
// ---------------------------------------------------------------------------

const VOWELS: &str = "aeıioöuüAEIİOÖUÜ";

pub fn turkish_lower(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            'I' => out.push('ı'),
            'İ' => out.push('i'),
            c => out.extend(c.to_lowercase()),
        }
    }
    out.replace('\u{307}', "")
}

const ONES: [&str; 10] = [
    "sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
];
const TENS: [&str; 10] = [
    "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
];

pub fn number_to_words(n: i64) -> String {
    if n < 0 {
        return format!("eksi {}", number_to_words(-n));
    }
    if n < 10 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let (t, o) = ((n / 10) as usize, (n % 10) as usize);
        return if o == 0 {
            TENS[t].to_string()
        } else {
            format!("{} {}", TENS[t], ONES[o])
        };
    }
    if n < 1000 {
        let (h, r) = ((n / 100) as usize, n % 100);
        let prefix = if h == 1 {
            "yüz".to_string()
        } else {
            format!("{} yüz", ONES[h])
        };
        return if r == 0 {
            prefix
        } else {
            format!("{prefix} {}", number_to_words(r))
        };
    }
    let scales: [(i64, &str); 4] = [
        (1_000_000_000_000, "trilyon"),
        (1_000_000_000, "milyar"),
        (1_000_000, "milyon"),
        (1_000, "bin"),
    ];
    for (scale, label) in scales {
        if n >= scale {
            let (count, rem) = (n / scale, n % scale);
            let prefix = if scale == 1_000 && count == 1 {
                label.to_string()
            } else {
                format!("{} {label}", number_to_words(count))
            };
            return if rem == 0 {
                prefix
            } else {
                format!("{prefix} {}", number_to_words(rem))
            };
        }
    }
    format!("cok-buyuk-{n}")
}

fn ordinal_exceptions(word: &str) -> Option<&'static str> {
    Some(match word {
        "bir" => "birinci",
        "iki" => "ikinci",
        "üç" => "üçüncü",
        "dört" => "dördüncü",
        "beş" => "beşinci",
        "altı" => "altıncı",
        "yedi" => "yedinci",
        "sekiz" => "sekizinci",
        "dokuz" => "dokuzuncu",
        "on" => "onuncu",
        "yirmi" => "yirminci",
        "otuz" => "otuzuncu",
        "kırk" => "kırkıncı",
        "elli" => "ellinci",
        "altmış" => "altmışıncı",
        "yetmiş" => "yetmişinci",
        "seksen" => "sekseninci",
        "doksan" => "doksanıncı",
        "yüz" => "yüzüncü",
        "bin" => "bininci",
        "milyon" => "milyonuncu",
        "milyar" => "milyarıncı",
        _ => return None,
    })
}

pub fn ordinal_to_words(n: i64) -> String {
    let cardinal = number_to_words(n);
    let mut words: Vec<&str> = cardinal.split(' ').collect();
    if let Some(last) = words.last_mut() {
        let rep = match ordinal_exceptions(last) {
            Some(s) => s.to_string(),
            None => format!("{last}inci"),
        };
        *last = Box::leak(rep.into_boxed_str());
    }
    words.join(" ")
}

fn digit_sequence_to_words(value: &str) -> String {
    value
        .chars()
        .filter_map(|c| c.to_digit(10).map(|d| ONES[d as usize]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn digit_group_to_words(value: &str) -> String {
    let stripped = value.trim_start_matches('0');
    let zeros = "sıfır ".repeat(value.len() - stripped.len());
    if stripped.is_empty() {
        return zeros.trim_end().to_string();
    }
    format!("{zeros}{}", number_to_words(stripped.parse::<i64>().unwrap_or(0))).trim().to_string()
}

fn decimal_to_words(value: &str) -> String {
    let mut parts = value.splitn(2, |c| c == ',' || c == '.');
    let whole = parts.next().unwrap_or("0");
    let fraction = parts.next().unwrap_or("");
    let whole_words = number_to_words(whole.parse::<i64>().unwrap_or(0));
    if fraction == "5" && whole.parse::<i64>().unwrap_or(0) != 0 {
        return format!("{whole_words} buçuk");
    }
    let frac_words = if fraction.len() <= 2 {
        digit_group_to_words(fraction)
    } else {
        digit_sequence_to_words(fraction)
    };
    format!("{whole_words} virgül {frac_words}")
}

fn numeric_to_words(value: &str) -> String {
    // "1.250" / "1.250,75" binlik ayracli; yoksa ondalik/tamsayi
    let grouped = re(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$");
    if grouped.is_match(value).unwrap_or(false) {
        let no_dots = value.replace('.', "");
        if let Some((w, f)) = no_dots.split_once(',') {
            return decimal_to_words(&format!("{w},{f}"));
        }
        return number_to_words(no_dots.parse::<i64>().unwrap_or(0));
    }
    if value.contains(',') || value.contains('.') {
        return decimal_to_words(value);
    }
    number_to_words(value.parse::<i64>().unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Harita ve yardimcilar
// ---------------------------------------------------------------------------

fn months(n: u32) -> Option<&'static str> {
    Some(match n {
        1 => "ocak",
        2 => "şubat",
        3 => "mart",
        4 => "nisan",
        5 => "mayıs",
        6 => "haziran",
        7 => "temmuz",
        8 => "ağustos",
        9 => "eylül",
        10 => "ekim",
        11 => "kasım",
        12 => "aralık",
        _ => return None,
    })
}

fn last_vowel_is_back(words: &str) -> bool {
    match words.chars().rev().find(|c| VOWELS.contains(*c)) {
        Some(v) => "aıou".contains(v),
        None => false,
    }
}

fn with_dative(words: &str) -> String {
    if let Some(stripped) = words.strip_suffix("dört") {
        return format!("{stripped}dörde");
    }
    // "dört" bitisli bilesik ("on dört" -> "on dörde"): son kelime kontrolu ustte;
    // genel kural:
    let vowel_back = last_vowel_is_back(words);
    let vowel = if vowel_back { 'a' } else { 'e' };
    let last = words.chars().last().unwrap_or(' ');
    let y = if VOWELS.contains(last) { "y" } else { "" };
    format!("{words}{y}{vowel}")
}

fn with_locative(words: &str) -> String {
    let vowel_back = last_vowel_is_back(words);
    let vowel = if vowel_back { 'a' } else { 'e' };
    let last = words.chars().last().unwrap_or(' ');
    let cons = if "çfhkpsşt".contains(last) { 't' } else { 'd' };
    format!("{words}{cons}{vowel}")
}

fn attach_suffix(stem: &str, suffix: &str) -> String {
    let mut stem = stem.to_string();
    let mut suffix = suffix.to_string();
    if !stem
        .chars()
        .last()
        .map(|c| VOWELS.contains(c))
        .unwrap_or(true)
        && suffix.len() > 1
    {
        let mut it = suffix.chars();
        let c0 = it.next().unwrap();
        let c1 = it.next().unwrap();
        if "yns".contains(c0) && VOWELS.contains(c1) {
            suffix = suffix[c0.len_utf8()..].to_string();
        }
    }
    if suffix.starts_with(|c: char| VOWELS.contains(c)) {
        if stem.ends_with("dört") {
            stem = format!("{}dörd", &stem[..stem.len() - "dört".len()]);
        } else if stem.ends_with("buçuk") {
            stem = format!("{}buçuğ", &stem[..stem.len() - "buçuk".len()]);
        }
    }
    let mut vowel = stem
        .chars()
        .rev()
        .find(|c| VOWELS.contains(*c))
        .unwrap_or('e');
    let mut out = String::new();
    for (index, char) in suffix.chars().enumerate() {
        let mut c = char;
        if "ae".contains(c) {
            c = if "aıou".contains(vowel) { 'a' } else { 'e' };
        } else if "ıiuü".contains(c) {
            c = match ("aıou".contains(vowel), "ouöü".contains(vowel)) {
                (true, false) => 'ı',
                (true, true) => 'u',
                (false, false) => 'i',
                (false, true) => 'ü',
            };
        } else if index == 0 && "dt".contains(c) {
            let last = stem.chars().last().unwrap_or(' ');
            c = if "çfhkpsşt".contains(last) { 't' } else { 'd' };
        }
        if VOWELS.contains(c) {
            vowel = c;
        }
        out.push(c);
    }
    format!("{stem}{out}")
}

fn letter_names_initialism() -> HashMap<char, &'static str> {
    // _LETTER_NAMES + K->ka, I->i (bas harf sozluk anahtari buyuk harf)
    let mut m: HashMap<char, &'static str> = [
        ('A', "a"), ('B', "be"), ('C', "ce"), ('Ç', "çe"), ('D', "de"),
        ('E', "e"), ('F', "fe"), ('G', "ge"), ('Ğ', "yumuşak ge"),
        ('H', "he"), ('I', "i"), ('İ', "i"), ('J', "je"), ('K', "ka"),
        ('L', "le"), ('M', "me"), ('N', "ne"), ('O', "o"), ('Ö', "ö"),
        ('P', "pe"), ('R', "re"), ('S', "se"), ('Ş', "şe"), ('T', "te"),
        ('U', "u"), ('Ü', "ü"), ('V', "ve"), ('Y', "ye"), ('Z', "ze"),
        ('Q', "kü"), ('W', "dabılyu"), ('X', "iks"),
    ]
    .into_iter()
    .collect();
    let _ = &mut m;
    m
}

fn letter_names_alpha() -> HashMap<char, &'static str> {
    let mut m = letter_names_initialism();
    m.insert('K', "ke");
    m.insert('I', "ı");
    m
}

fn abbreviations() -> Vec<(&'static str, &'static str)> {
    vec![
        ("dr.", "doktor"), ("prof.", "profesör"), ("vb.", "ve benzeri"),
        ("vs.", "vesaire"), ("örn.", "örneğin"), ("no.", "numara"),
        ("mg", "miligram"), ("kg", "kilogram"), ("gr", "gram"),
        ("km", "kilometre"), ("cm", "santimetre"), ("mm", "milimetre"),
        ("lt", "litre"), ("ml", "mililitre"), ("kb", "kilobayt"),
        ("mb", "megabayt"), ("gb", "gigabayt"), ("tb", "terabayt"),
        ("tl", "lira"), ("doç.", "doçent"), ("uzm.", "uzman"),
        ("op.", "operatör"), ("yrd.", "yardımcı"),
        ("öğr. üyesi", "öğretim üyesi"), ("öğr. gör.", "öğretim görevlisi"),
        ("dt.", "diş hekimi"), ("dyt.", "diyetisyen"), ("psk.", "psikolog"),
        ("fzt.", "fizyoterapist"), ("hemş.", "hemşire"), ("ecz.", "eczacı"),
        ("av.", "avukat"), ("müh.", "mühendis"), ("hz.", "hazreti"),
        ("bkz.", "bakınız"), ("vd.", "ve diğerleri"),
        ("mah.", "mahallesi"), ("cad.", "caddesi"), ("cd.", "caddesi"),
        ("sok.", "sokak"), ("sk.", "sokak"), ("bul.", "bulvarı"),
        ("blv.", "bulvarı"), ("apt.", "apartmanı"),
        ("ltd. şti.", "limited şirketi"), ("şti.", "şirketi"),
    ]
}

fn before_a_word(abbr: &str) -> bool {
    matches!(
        abbr, "doç." | "uzm." | "op." | "yrd." | "dt." | "dyt." | "psk."
        | "fzt." | "hemş." | "ecz." | "av." | "müh." | "hz." | "mah."
        | "cad." | "cd." | "sok." | "sk." | "bul." | "blv." | "apt."
    )
}

fn caps_lexicon(word: &str) -> Option<&'static str> {
    Some(match word {
        "MR" => "emar", "SMS" => "esemes", "NTV" => "entivi", "YÖK" => "yök",
        "BİM" => "bim", "ŞOK" => "şok", "COVID" => "kovid", "MEB" => "meb",
        "İTÜ" => "itü", "IBAN" => "iban", "İETT" => "ietete", "ALO" => "alo",
        "MİT" => "mit", "OK" => "okey", "TSH" => "teseha", "MHRS" => "meharese",
        "HES" => "hes", "CEO" => "siio", "PR" => "piar", "DJ" => "dijey",
        "TV" => "tivi", "FBI" => "efbiay", "CNN" => "sienen",
        "VIP" => "viaypi", "HD" => "heyçdi",
        _ => return None,
    })
}

fn address_words(word: &str) -> Option<&'static str> {
    Some(match word {
        "www" => "dabılyu dabılyu dabılyu",
        "com" => "kom", "net" => "net", "org" => "org", "gov" => "gov",
        "info" => "info", "gmail" => "cimeyl", "hotmail" => "hotmeyl",
        "outlook" => "autluk", "yahoo" => "yahu", "icloud" => "ayklaud",
        _ => return None,
    })
}

fn number_units(unit: &str) -> Option<&'static str> {
    Some(match unit {
        "m" => "metre", "g" => "gram", "cc" => "sisi", "mcg" => "mikrogram",
        "μg" => "mikrogram", "IU" => "ünite", "kcal" => "kilokalori",
        "mmHg" => "milimetre cıva", "dk" => "dakika", "sn" => "saniye",
        "sa" => "saat", "ms" => "milisaniye", "dl" => "desilitre",
        _ => return None,
    })
}

fn is_pronounceable(word: &str) -> bool {
    let len = word.chars().count();
    if len < 4 {
        return false;
    }
    let vowels = word.chars().filter(|c| VOWELS.contains(*c)).count();
    if (vowels as f32) < 0.3 * len as f32 {
        return false;
    }
    // uc sessiz kosusu yok
    let mut run = 0;
    for ch in word.chars() {
        if "AEIİOÖUÜ".contains(ch) {
            run = 0;
        } else {
            run += 1;
            if run >= 3 {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Ana boru hatti (normalize_for_model + cased/model son adimlari, birebir sirayla)
// NOT: foreign.py (cmudict Ingilizce/marka) v1'de YOK; kelime oldugu gibi gecer.
// ---------------------------------------------------------------------------

fn esc_dot(s: &str) -> String {
    s.replace('.', "\\.")
}

fn replace_currency_words(symbol: &str, whole: &str, fraction: Option<&str>) -> String {
    let (unit, subunit) = match symbol {
        "₺" => ("lira", "kuruş"),
        "$" => ("dolar", "sent"),
        "€" => ("avro", "sent"),
        "£" => ("sterlin", "peni"),
        _ => ("lira", "kuruş"),
    };
    let whole_n: i64 = whole.replace('.', "").parse().unwrap_or(0);
    let mut value = format!("{} {unit}", number_to_words(whole_n));
    if let Some(f) = fraction {
        let f2 = format!("{f:0<2}");
        if f2.parse::<i64>().unwrap_or(0) != 0 {
            value += &format!(
                " {} {subunit}",
                number_to_words(f2.parse::<i64>().unwrap_or(0))
            );
        }
    }
    value
}

fn replace_phone_caps(caps: &Captures) -> String {
    let g = |i: usize| caps.get(i).map(|m| m.as_str()).unwrap_or("");
    let country = g(1);
    let trunk = g(2);
    let groups: Vec<&str> = (3..=6).map(|i| g(i)).collect();
    let mut spoken: Vec<String> = Vec::new();
    if !country.is_empty() {
        spoken.push("artı doksan".to_string());
    }
    if !trunk.is_empty() {
        spoken.push(format!("sıfır {}", digit_group_to_words(groups[0])));
    } else {
        spoken.push(digit_group_to_words(groups[0]));
    }
    for grp in groups.iter().skip(1) {
        spoken.push(digit_group_to_words(grp));
    }
    spoken.join(", ")
}

fn replace_range_caps(caps: &Captures) -> String {
    let start = caps.get(1).map(|m| m.as_str()).unwrap_or("");
    let end = caps.get(2).map(|m| m.as_str()).unwrap_or("");
    let season = start.len() == 4
        && start.chars().all(|c| c.is_ascii_digit())
        && end.len() == 2
        && end.chars().all(|c| c.is_ascii_digit())
        && end.parse::<i64>().unwrap_or(-1) == (start.parse::<i64>().unwrap_or(-99) + 1) % 100;
    if season {
        return format!("{start} {}", end.parse::<i64>().unwrap_or(0));
    }
    let both_digits =
        !start.is_empty() && !end.is_empty() && start.chars().all(|c| c.is_ascii_digit()) && end.chars().all(|c| c.is_ascii_digit());
    if both_digits {
        let (s, e) = (start.parse::<i64>().unwrap_or(0), end.parse::<i64>().unwrap_or(0));
        if s > e {
            return format!("{start} {end}");
        }
    }
    format!("{start} ile {end}")
}

fn replace_iban_caps(caps: &Captures) -> String {
    let g2 = caps.get(2).map(|m| m.as_str()).unwrap_or("");
    let mut groups = vec![caps.get(1).map(|m| m.as_str()).unwrap_or("")];
    if g2.contains(' ') {
        groups.extend(g2.split(' '));
    } else {
        groups.push(g2);
    }
    groups.push(caps.get(3).map(|m| m.as_str()).unwrap_or(""));
    let parts: Vec<String> = groups.iter().map(|g| digit_sequence_to_words(g)).collect();
    format!("te re, {}", parts.join(", "))
}

fn replace_times_caps(caps: &Captures, full: &str) -> String {
    let daily = caps.get(1).map(|m| m.as_str()).unwrap_or("");
    let end = caps.get(0).map(|m| m.end()).unwrap_or(0);
    let first = number_to_words(caps.get(2).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0));
    let second = number_to_words(caps.get(3).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0));
    let dose_words = ["tablet", "kapsül", "şurup", "ölçek", "damla", "doz", "puf",
        "kaşık", "ampul", "draje", "saşe", "fitil", "iğne"];
    let rest = turkish_lower(&full[end.min(full.len())..]);
    let is_dose = dose_words.iter().any(|w| rest.trim_start().starts_with(*w));
    if !daily.is_empty() || is_dose {
        return format!("{daily}{first} defa {second}");
    }
    format!("{} {}", with_dative(&first), second)
}

fn replace_lone_letter(full: &str, caps: &Captures) -> String {
    let m0 = caps.get(0).map(|m| m.as_str()).unwrap_or("");
    let letter = m0.chars().next().unwrap_or(' ');
    let start = caps.get(0).map(|m| m.start()).unwrap_or(0);
    let end = caps.get(0).map(|m| m.end()).unwrap_or(0);
    let following = full[end..].chars().next();
    if letter == 'O' || letter == 'I' {
        return m0.to_string();
    }
    if (letter == 'A' || letter == 'E') && matches!(following, Some(',') | Some('!') | Some('?')) {
        return m0.to_string();
    }
    let initial = caps.get(2).map(|m| !m.as_str().is_empty()).unwrap_or(false) || following == Some('.');
    let before = &full[..start.min(full.len())];
    let after = &full[end.min(full.len())..];
    let before_low = turkish_lower(before);
    let vitamin = before_low
        .split_whitespace()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
        .contains("vitamin")
        || after.trim_start().to_lowercase().starts_with("vitamin");
    let table = if initial || vitamin {
        letter_names_initialism()
    } else {
        letter_names_alpha()
    };
    table.get(&letter).copied().unwrap_or("").to_string()
}

fn replace_initialism_caps(caps: &Captures) -> String {
    let word = caps.get(0).map(|m| m.as_str()).unwrap_or("");
    if let Some(exp) = caps_lexicon(word) {
        return exp.to_string();
    }
    if is_pronounceable(word) {
        return word.to_string();
    }
    let table = letter_names_initialism();
    word.chars()
        .map(|c| table.get(&c).copied().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("")
}

fn replace_slash_pair_caps(caps: &Captures) -> String {
    let (num, den) = (
        caps.get(1).map(|m| m.as_str()).unwrap_or("0").parse::<i64>().unwrap_or(0),
        caps.get(2).map(|m| m.as_str()).unwrap_or("1").parse::<i64>().unwrap_or(1),
    );
    if num > 0 && num < den && den <= 10 {
        return format!("{} {}", with_locative(&number_to_words(den)), number_to_words(num));
    }
    if (num == 7 && den == 24) || (num == 24 && den == 7) {
        return format!("{} {}", number_to_words(num), number_to_words(den));
    }
    format!("{} bölü {}", number_to_words(num), number_to_words(den))
}

fn replace_address_caps(caps: &Captures) -> String {
    let mut address = caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
    let low_head = address.to_lowercase();
    for prefix in ["https://", "http://"] {
        if low_head.starts_with(prefix) {
            address = address[prefix.len()..].to_string();
            break;
        }
    }
    while address.ends_with('/') {
        address.pop();
    }
    let mut spoken: Vec<String> = Vec::new();
    // ([.@/_-]) ile bol
    let mut cur = String::new();
    let mut parts: Vec<String> = Vec::new();
    for ch in address.chars() {
        if ".@/_-".contains(ch) {
            if !cur.is_empty() {
                parts.push(cur.clone());
                cur.clear();
            }
            parts.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    for piece in parts {
        match piece.as_str() {
            "." => spoken.push("nokta".to_string()),
            "@" => spoken.push("et".to_string()),
            "/" => spoken.push("bölü".to_string()),
            "_" => spoken.push("alt tire".to_string()),
            "-" => spoken.push("tire".to_string()),
            _ => {
                let lower = turkish_lower(&piece);
                if let Some(w) = address_words(&lower) {
                    spoken.push(w.to_string());
                } else if lower.chars().count() <= 2 && lower.chars().all(|c| c.is_alphabetic()) {
                    let names = letter_names_alpha();
                    let spelled: Vec<&str> = lower
                        .chars()
                        .map(|c| {
                            let up = match c {
                                'i' => 'İ',
                                'ı' => 'I',
                                _ => c.to_uppercase().next().unwrap_or(c),
                            };
                            names.get(&up).copied().unwrap_or("")
                        })
                        .collect();
                    spoken.push(spelled.join(" "));
                } else {
                    spoken.push(piece);
                }
            }
        }
    }
    spoken.join(" ")
}

fn weekday_full(short: &str) -> Option<&'static str> {
    Some(match short {
        "Pzt" => "pazartesi",
        "Sal" => "salı",
        "Çar" => "çarşamba",
        "Per" => "perşembe",
        "Cum" => "cuma",
        "Cmt" => "cumartesi",
        "Paz" => "pazar",
        _ => return None,
    })
}

const ORDINAL_HEADS: &str = "Dünya|Lig|Ordu|Kolordu|Tümen|Dönem|Meclis|Kurultay|Kongre|Uluslararası|Ulusal|Olağan|Olağanüstü|Geleneksel|Yıl|Yüzyıl|Sınıf|Kat|Blok|Etap|Bölge|Asliye|Ağır|Sulh|İcra|Aile|Selim|Murat|Murad|Mehmet|Mehmed|Mahmut|Mahmud|Ahmet|Ahmed|Osman|Bayezid|Beyazıt|Süleyman|Mustafa|Abdülhamid|Abdülmecid|Louis|Henry|Napolyon";

fn replace_date_parts(day_text: &str, month_text: &str, year_text: &str, original: Option<&str>) -> String {
    let day: u32 = day_text.parse().unwrap_or(0);
    let month: u32 = month_text.parse().unwrap_or(0);
    let year: i64 = year_text.parse().unwrap_or(0);
    match months(month) {
        Some(m) if (1..=31).contains(&day) => {
            format!("{} {m} {}", number_to_words(day as i64), number_to_words(year))
        }
        _ => original.unwrap_or("").to_string(),
    }
}

fn replace_time_caps(caps: &Captures) -> String {
    let hour: i64 = caps.get(1).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0);
    let minute = caps.get(2).map(|m| m.as_str()).unwrap_or("0");
    let hour_words = number_to_words(hour);
    if minute.parse::<i64>().unwrap_or(-1) == 0 {
        return hour_words;
    }
    let minute_words = if minute.starts_with('0') {
        digit_sequence_to_words(minute)
    } else {
        number_to_words(minute.parse().unwrap_or(0))
    };
    format!("{hour_words} {minute_words}")
}

fn replace_duration_caps(caps: &Captures) -> String {
    let nums: Vec<i64> = (1..=3)
        .map(|i| caps.get(i).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0))
        .collect();
    let mut parts: Vec<String> = Vec::new();
    for (value, unit) in [(nums[0], "saat"), (nums[1], "dakika")] {
        if value != 0 {
            parts.push(format!("{} {unit}", number_to_words(value)));
        }
    }
    if nums[2] != 0 || parts.is_empty() {
        parts.push(format!("{} saniye", number_to_words(nums[2])));
    }
    parts.join(" ")
}

/// Ham metin -> konusulur metin (normalize_for_model + cased son adimlar).
/// NOT: foreign.py (cmudict Ingilizce/marka) v1'de YOK; o kelimeler aynen gecer.
pub fn normalize_for_model_text(raw: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    // NFC (to_model_text girisi)
    let mut t: String = raw.nfc().collect();
    t = normalize_for_model(&t);
    // CAPS_WORD -> turkish_lower
    let caps_re = re(r"\b[A-ZÇĞİÖŞÜQWX]{2,}\b");
    t = caps_re
        .replace_all(&t, |caps: &Captures| turkish_lower(caps.get(0).map(|m| m.as_str()).unwrap_or("")))
        .into_owned();
    // I->ı, İ->i (diger buyukler kalir)
    t = t.chars().map(|c| match c { 'I' => 'ı', 'İ' => 'i', _ => c }).collect();
    // to_model_text sonu: turkish_lower + clean
    clean_model_text(&turkish_lower(&t))
}

fn clean_model_text(t: &str) -> String {
    // _FOLD: tirnak varyantlari SILINIR (None)
    let mut s = String::with_capacity(t.len());
    for ch in t.chars() {
        match ch {
            '"' | '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{00ab}' | '\u{00bb}' => {}
            _ => s.push(ch),
        }
    }
    // alfabe disi -> bosluk
    let vocab_keep = |c: char| {
        c.is_whitespace()
            || "abcçdefgğhıijklmnoöpqrsştuüvwxyzâîû .,?!'-:;".contains(c)
    };
    let s2: String = s.chars().map(|c| if vocab_keep(c) { c } else { ' ' }).collect();
    let s3 = SP_PUNCT_RE
        .get_or_init(|| re(r"\s+([.,?!:;])"))
        .replace_all(&s2, "$1")
        .into_owned();
    collapse_spaces(&s3)
}

fn normalize_for_model(text: &str) -> String {
    let mut t = text.replace('\u{2212}', "-");
    t = normalize_orthography(&t);
    t = lower_if_shouted(&t);
    t = replace_roman_text(&t);
    // "ABD'DE": kesme sonrasi buyuk kisaltma kucultulur
    t = re(r"'([A-ZÇĞİÖŞÜ]+)\b")
        .replace_all(&t, |c: &Captures| {
            format!("'{}", turkish_lower(c.get(1).map(|m| m.as_str()).unwrap_or("")))
        })
        .into_owned();
    // "Kimlik No" vs "No: 12"
    t = rei(r"(?<!\bsokak )(?<!\bsokağı )(?<!\bcadde )(?<!\bcaddesi )(?<!\bbulvarı )(?<!\bmahallesi )(?<!\bapartmanı )(?<=[^\W\d_] )no\b\.?:?(?=\s)")
        .replace_all(&t, "numarası")
        .into_owned();
    t = rei(r"\bno\s*[:.]\s*(?=\d)")
        .replace_all(&t, "numara ")
        .into_owned();
    t = re(r"\bD\s*:\s*(?=\d)").replace_all(&t, "daire ").into_owned();
    t = re(r"\bTel\s*:").replace_all(&t, "telefon:").into_owned();
    t = re(r"\bs\.\s*(?=\d)").replace_all(&t, "sayfa ").into_owned();
    t = replace_emails_urls(&t);
    t = replace_symbols_text(&t);
    // IBAN
    t = re(r"\bTR(\d{2})((?:\s?\d{4}){5})\s?(\d{2})\b")
        .replace_all(&t, |c: &Captures| replace_iban_caps(&c))
        .into_owned();
    // Noktali kisaltma "T.C."
    t = re(r"\b(?:[A-ZÇĞİÖŞÜ]\.){2,}")
        .replace_all(&t, |c: &Captures| {
            c.get(0).map(|m| m.as_str()).unwrap_or("").chars().filter(|&ch| ch != '.').map(|ch| {
                letter_names_initialism().get(&ch).copied().unwrap_or("")
            }).collect::<Vec<_>>().join("")
        })
        .into_owned();
    t = re(r"\bpH\b").replace_all(&t, "pe ha").into_owned();
    t = re(r"\bH2O\b").replace_all(&t, "haş iki o").into_owned();
    t = rei(r"\bH\.\s*pylori\b").replace_all(&t, "ha pilori").into_owned();
    t = rei(r"\bX-?ray\b").replace_all(&t, "iksrey").into_owned();
    t = re(r"\b(hane(?:si|leri)?)\s+(\d{2,6})\b")
        .replace_all(&t, |c: &Captures| {
            format!("{} {}", c.get(1).map(|m| m.as_str()).unwrap_or(""), digit_sequence_to_words(c.get(2).map(|m| m.as_str()).unwrap_or("")))
        })
        .into_owned();
    t = re(r"\bHbA1c\b").replace_all(&t, "ha be a bir ce").into_owned();
    t = re(r"\b(AB|A|B|0)\s?Rh\s?\(?([+-])\)?")
        .replace_all(&t, |c: &Captures| {
            let grp = c.get(1).map(|m| m.as_str()).unwrap_or("");
            let g = if grp == "0" {
                "sıfır".to_string()
            } else {
                grp.chars().map(|ch| letter_names_initialism().get(&ch).copied().unwrap_or("")).collect::<Vec<_>>().join("")
            };
            format!("{g} eraj {}", if c.get(2).map(|m| m.as_str()) == Some("+") { "pozitif" } else { "negatif" })
        })
        .into_owned();
    // Tansiyon 120/80
    t = re(r"\b(\d{2,3})/(\d{2,3})\s*mmHg\b")
        .replace_all(&t, |c: &Captures| {
            let a = number_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0));
            let b = number_to_words(c.get(2).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0));
            format!("{} {b} milimetre cıva", with_dative(&a))
        })
        .into_owned();
    // Doz: günde 2x1 / boyut 30x40
    t = {
        let r = re(r"(\bgünde\s+)?\b(\d+)\s?[xX×]\s?(\d+)\b");
        let full = t.clone();
        r.replace_all(&t, |c: &Captures| replace_times_caps(&c, &full)).into_owned()
    };
    // Hiz: 90 km/sa
    t = re(r"(?<![\d.,])(\d+(?:,\d+)?)\s?km/(?:sa|s|h)\b")
        .replace_all(&t, |c: &Captures| {
            format!("saatte {} kilometre", numeric_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0")))
        })
        .into_owned();
    // Doz agirlik: 5 mg/kg
    t = re(r"(?<![\d.,])(\d+(?:,\d+)?)\s?(mg|mcg|g|ml)/kg\b")
        .replace_all(&t, |c: &Captures| {
            let units = [("mg", "miligram"), ("mcg", "mikrogram"), ("g", "gram"), ("ml", "mililitre")];
            let u = c.get(2).map(|m| m.as_str()).unwrap_or("mg");
            let uw = units.iter().find(|(k, _)| *k == u).map(|(_, w)| *w).unwrap_or(u);
            format!("kilogram başına {} {uw}", numeric_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0")))
        })
        .into_owned();
    // Alan/hacim: m2, cm3 (NFKC sonrasi)
    t = re(r"(?<=\d) ?(km|cm|m)([23])(?!\w)")
        .replace_all(&t, |c: &Captures| {
            let base = match c.get(1).map(|m| m.as_str()).unwrap_or("m") {
                "km" => "kilometre", "cm" => "santimetre",
                _ => "metre",
            };
            format!(" {base}{}", if c.get(2).map(|m| m.as_str()) == Some("2") { "kare" } else { "küp" })
        })
        .into_owned();
    // Sayi+birim: 5 kg, 3 dk
    {
        let units = ["m", "g", "cc", "mcg", "μg", "IU", "kcal", "mmHg", "dk", "sn", "sa", "ms", "dl"];
        let mut alts: Vec<String> = units.iter().map(|u| regex_escape(u)).collect();
        alts.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let pat = format!(r"(?<=\d) ?({})(?![\w/])", alts.join("|"));
        t = re(&pat)
            .replace_all(&t, |c: &Captures| {
                let u = c.get(1).map(|m| m.as_str()).unwrap_or("");
                format!(" {}", number_units(u).unwrap_or(u))
            })
            .into_owned();
    }
    // Litre: 2 L su
    t = re(r"(?<=\d) L\b(?![/'’])(?!\s*\d)")
        .replace_all(&t, " litre")
        .into_owned();
    // Hafta gunleri + araliklari
    t = rei(r"\b(Pzt|Sal|Çar|Per|Cum|Cmt|Paz)\b\.?(?=\s*(?:[-–,/]|\d|$))")
        .replace_all(&t, |c: &Captures| {
            weekday_full(c.get(1).map(|m| m.as_str()).unwrap_or("")).unwrap_or("").to_string()
        })
        .into_owned();
    t = rei(r"\b(pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar)\s*[-–]\s*(pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar)\b")
        .replace_all(&t, "$1 ile $2")
        .into_owned();
    // Para: ₺12,50 ve 12,50 TL
    t = re(r"([₺$€£])\s*((?:\d{1,3}(?:\.\d{3})+)|\d+)(?:[,.](\d{1,2}))?")
        .replace_all(&t, |c: &Captures| {
            replace_currency_words(
                c.get(1).map(|m| m.as_str()).unwrap_or("₺"),
                c.get(2).map(|m| m.as_str()).unwrap_or("0"),
                {
                    let f = c.get(3).map(|m| m.as_str()).unwrap_or("");
                    if f.is_empty() { None } else { Some(f) }
                },
            )
        })
        .into_owned();
    t = re(r"(?<![\d.,])((?:\d{1,3}(?:\.\d{3})+)|\d+)(?:,(\d{1,2}))?\s?(?:([₺$€£])|(TL|USD|EUR|GBP)\b)")
        .replace_all(&t, |c: &Captures| {
            let sym = c.get(3).map(|m| m.as_str()).unwrap_or("");
            let code = c.get(4).map(|m| m.as_str()).unwrap_or("");
            let symbol = if !sym.is_empty() {
                sym
            } else {
                match code {
                    "TL" => "₺", "USD" => "$", "EUR" => "€", "GBP" => "£",
                    _ => "₺",
                }
            };
            replace_currency_words(
                symbol,
                c.get(1).map(|m| m.as_str()).unwrap_or("0"),
                {
                    let f = c.get(2).map(|m| m.as_str()).unwrap_or("");
                    if f.is_empty() { None } else { Some(f) }
                },
            )
        })
        .into_owned();
    // Kod harfleri: 52EB4297
    t = re(r"(?<=\d)[A-ZÇĞİÖŞÜQWX]{1,6}(?=\d|\b)|\b[A-ZÇĞİÖŞÜQWX]{1,6}(?=\d)|(?<=\d )[A-ZÇĞİÖŞÜQWX]{2,6}(?= \d)")
        .replace_all(&t, |c: &Captures| {
            let word = c.get(0).map(|m| m.as_str()).unwrap_or("");
            if caps_lexicon(word).is_some() || is_pronounceable(word) {
                return word.to_string();
            }
            let table = letter_names_initialism();
            format!(" {} ", word.chars().map(|ch| table.get(&ch).copied().unwrap_or("")).collect::<Vec<_>>().join(" "))
        })
        .into_owned();
    // Harf-sayi ayirma: A4, 5G
    t = re(r"(?<=[^\W\d_])(?=\d)").replace_all(&t, " ").into_owned();
    t = re(r"(?<=\d)(?=[^\W\d_])").replace_all(&t, " ").into_owned();
    t = re(r"(?<=[^\W\d_])-(?=\d)").replace_all(&t, " ").into_owned();
    // Sayi yanindaki tek harf
    t = {
        let r = re(r"\b([A-ZÇĞİÖŞÜQWX])(?= \d)|(?<=\d )([A-ZÇĞİÖŞÜQWX])\b");
        let full = t.clone();
        r.replace_all(&t, |c: &Captures| replace_lone_letter(&full, &c)).into_owned()
    };
    // Tarihler
    t = re(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
        .replace_all(&t, |c: &Captures| {
            replace_date_parts(
                c.get(1).map(|m| m.as_str()).unwrap_or(""),
                c.get(2).map(|m| m.as_str()).unwrap_or(""),
                c.get(3).map(|m| m.as_str()).unwrap_or(""),
                c.get(0).map(|m| m.as_str()),
            )
        })
        .into_owned();
    t = re(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
        .replace_all(&t, |c: &Captures| {
            replace_date_parts(
                c.get(3).map(|m| m.as_str()).unwrap_or(""),
                c.get(2).map(|m| m.as_str()).unwrap_or(""),
                c.get(1).map(|m| m.as_str()).unwrap_or(""),
                None,
            )
        })
        .into_owned();
    // Yuzde
    t = re(r"%\s*(\d+(?:[,.]\d+)?)")
        .replace_all(&t, |c: &Captures| {
            format!("yüzde {}", numeric_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0")))
        })
        .into_owned();
    t = re(r"(?<![\d.,])(\d+(?:,\d+)?)\s?%")
        .replace_all(&t, |c: &Captures| {
            format!("yüzde {}", numeric_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0")))
        })
        .into_owned();
    // Telefon + aralik + eksi
    t = re(r"(?<!\d)(\+90[\s-]?)?(0?)\(?(\d{3})\)?[\s-]?(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})(?!\d)")
        .replace_all(&t, |c: &Captures| replace_phone_caps(&c))
        .into_owned();
    t = re(r"(?<![\d\-–])(\d[\d.:,]*)\s*[-–]\s*(\d[\d.:,]*)(?![\d\-–])")
        .replace_all(&t, |c: &Captures| replace_range_caps(&c))
        .into_owned();
    t = re(r"(?<![\w.,:)\]])-(?=\d)").replace_all(&t, "eksi ").into_owned();
    // Sure + saat
    t = re(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)\b")
        .replace_all(&t, |c: &Captures| replace_duration_caps(&c))
        .into_owned();
    t = re(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
        .replace_all(&t, |c: &Captures| replace_time_caps(&c))
        .into_owned();
    // Kesir ciftleri
    t = re(r"\b(\d{1,2})/(\d{1,2})\b")
        .replace_all(&t, |c: &Captures| replace_slash_pair_caps(&c))
        .into_owned();
    t = re(r"\b(\d+)/(\d+)\b")
        .replace_all(&t, "$1 bölü $2")
        .into_owned();
    // Binlik + ondalik
    t = re(r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b")
        .replace_all(&t, |c: &Captures| numeric_to_words(c.get(0).map(|m| m.as_str()).unwrap_or("0")))
        .into_owned();
    t = re(r"\b(\d+)[,.](\d+)\b")
        .replace_all(&t, |c: &Captures| decimal_to_words(c.get(0).map(|m| m.as_str()).unwrap_or("0")))
        .into_owned();
    // Sira sayilari (3 yol)
    t = re(r"\b(\d+)'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncü|ncu)\b")
        .replace_all(&t, |c: &Captures| ordinal_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0)))
        .into_owned();
    t = re(r"\b(\d+)\.(?=[a-zçğıöşü])")
        .replace_all(&t, |c: &Captures| ordinal_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0)))
        .into_owned();
    t = re(&format!(r"\b([1-9]\d*)\.(?=\s+(?:[a-zçğıöşü]|(?:{ORDINAL_HEADS})\b))"))
        .replace_all(&t, |c: &Captures| ordinal_to_words(c.get(1).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0)))
        .into_owned();
    // Uzun ve sifirli sayi dizileri + kalan tamsayilar
    t = re(r"\b\d{5,}\b")
        .replace_all(&t, |c: &Captures| digit_sequence_to_words(c.get(0).map(|m| m.as_str()).unwrap_or("")))
        .into_owned();
    t = re(r"\b0\d+\b")
        .replace_all(&t, |c: &Captures| digit_group_to_words(c.get(0).map(|m| m.as_str()).unwrap_or("")))
        .into_owned();
    t = re(r"\b\d+\b")
        .replace_all(&t, |c: &Captures| number_to_words(c.get(0).map(|m| m.as_str()).unwrap_or("0").parse().unwrap_or(0)))
        .into_owned();
    // Oranlar + sayin
    t = rei(r"\bno\.'?s[ıu]\b").replace_all(&t, "numarası").into_owned();
    t = re(r"(?<=\w)/(sn|sa|dk)\b")
        .replace_all(&t, |c: &Captures| {
            let u = c.get(1).map(|m| m.as_str()).unwrap_or("").to_lowercase();
            let w = match u.as_str() {
                "sn" => "saniye", "sa" => "saat",
                _ => "dakika",
            };
            format!(" bölü {w}")
        })
        .into_owned();
    t = re(r"\b(mg|kg|g|km|m|cm|ml)/(kg|g|m|l|lt|ml|saat|gün)\b")
        .replace_all(&t, "$1 bölü $2")
        .into_owned();
    t = re(r"\bSn\.\s*(?=[A-ZÇĞİÖŞÜ])").replace_all(&t, "sayın ").into_owned();
    // Kisaltmalar (uzunluga gore azalan)
    {
        let mut abbrs = abbreviations();
        abbrs.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
        for (abbr, expansion) in abbrs {
            let esc = esc_dot(abbr);
            if abbr.ends_with('.') {
                let pat = format!(r"(?<!\w){esc}\s*(?=\w)");
                t = rei(&pat).replace_all(&t, format!("{expansion} ")).into_owned();
                if before_a_word(abbr) {
                    continue;
                }
                let pat2 = format!(r"(?<!\w){esc}");
                t = rei(&pat2).replace_all(&t, expansion).into_owned();
            } else {
                let pat = format!(r"(?<!\w){esc}(?!\w)");
                t = rei(&pat).replace_all(&t, expansion).into_owned();
            }
        }
    }
    // Buyuk harfli kisaltmalar + yalniz buyuk harfler
    t = re(r"\b[A-ZÇĞİÖŞÜQWX]{2,6}\b")
        .replace_all(&t, |c: &Captures| replace_initialism_caps(&c))
        .into_owned();
    t = {
        let r = re(r"(?<![\w.'-])([A-ZÇĞİÖŞÜQWX])(\.(?=\s+[a-zçğıöşü]))?(?![\w-])");
        let full = t.clone();
        r.replace_all(&t, |c: &Captures| replace_lone_letter(&full, &c)).into_owned()
    };
    // Ek uyumu + egik cizgi/noktalama temizligi
    t = replace_stem_suffix(&t);
    t = re(r"\s+/\s+").replace_all(&t, ", ").into_owned();
    t = re(r"(?<=[^\W\d_])/(?=[^\W\d_])").replace_all(&t, " ").into_owned();
    t = re(r"\s+([,.;:!?])").replace_all(&t, "$1").into_owned();
    t = re(r"^[\s,]+|(?<=[,.;:!?])\s*,").replace_all(&t, "").into_owned();
    collapse_spaces(&t.trim().to_string())
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ".+*?()|[]{}^$\\".contains(ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn replace_symbols_text(t: &str) -> String {
    let mut s = t.to_string();
    // derece: 25°C -> 25 derece
    s = re(r"(?<=\d)\s?°\s?C?(?![^\W\d_])")
        .replace_all(&s, " derece")
        .into_owned();
    s = re(r"(?<=\d)\s?\+\s?(?=\d)|(?<=\s)\+(?=\s)")
        .replace_all(&s, " artı ")
        .into_owned();
    s = re(r"(?<=[\d\s])=(?=[\d\s])").replace_all(&s, " eşittir ").into_owned();
    s = re(r"(?<![\w&])#(?=[^\W\d_])").replace_all(&s, "heşteg ").into_owned();
    s = re(r"\*+|_{2,}").replace_all(&s, "").into_owned();
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str(" ve "),
            '•' | '▪' | '►' | '→' | '⇒' => out.push_str(" , "),
            _ => out.push(ch),
        }
    }
    // So/Cs/Co kategorileri: yaklasik kural — alfanumerik/bosluk/noktalama disi atilir
    const KEEP_PUNCT: &str = " .,?!'\"-_:;()[]…–—«»„“”‘’ʼ´`@/_+=%₺$€£&#°^*~";
    let out2: String = out
        .chars()
        .filter(|&c| c.is_alphanumeric() || c.is_whitespace() || KEEP_PUNCT.contains(c))
        .collect();
    out2.replace('\u{fe0f}', "").replace('\u{200d}', "")
}

fn replace_emails_urls(t: &str) -> String {
    let mut s = t.to_string();
    s = re(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
        .replace_all(&s, |c: &Captures| replace_address_caps(&c))
        .into_owned();
    s = rei(r"\b(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|gov|edu|info|io|ai|tr)\b(?:/[\w./-]*)?")
        .replace_all(&s, |c: &Captures| replace_address_caps(&c))
        .into_owned();
    s
}

fn replace_roman_text(t: &str) -> String {
    let mut s = t.to_string();
    let roman = r"(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|IX|VIII|VII|VI|IV|III|II)";
    s = re(&format!(r"\b({roman}|I(?=\.\s+Dünya))\.\s+(?=[A-ZÇĞİÖŞÜ])"))
        .replace_all(&s, |c: &Captures| {
            format!("{} ", ordinal_to_words(roman_to_int(c.get(1).map(|m| m.as_str()).unwrap_or("I"))))
        })
        .into_owned();
    s = re(&format!(r"\b([A-ZÇĞİÖŞÜ][^\W\d_]*\s+)({roman})\b(?![.'’]?[^\W\d_])"))
        .replace_all(&s, |c: &Captures| {
            format!(
                "{}{}",
                c.get(1).map(|m| m.as_str()).unwrap_or(""),
                number_to_words(roman_to_int(c.get(2).map(|m| m.as_str()).unwrap_or("I")))
            )
        })
        .into_owned();
    s
}

fn roman_to_int(numeral: &str) -> i64 {
    let val = |c: char| match c {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        _ => 0,
    };
    let chars: Vec<char> = numeral.chars().collect();
    let mut total = 0i64;
    for (i, &ch) in chars.iter().enumerate() {
        let v = val(ch);
        let next = chars.get(i + 1).map(|c| val(*c)).unwrap_or(0);
        if next > v {
            total -= v;
        } else {
            total += v;
        }
    }
    total
}

fn replace_stem_suffix(t: &str) -> String {
    // Konusulan kokler (uzunluga gore azalan) + sira eki alternatifi
    let mut stems: Vec<String> = vec![
        "yüz", "bin", "milyon", "milyar", "trilyon", "lira", "kuruş", "dolar",
        "avro", "sent", "sterlin", "peni", "saniye", "dakika", "saat", "buçuk",
        "metrekare", "metreküp", "kilometrekare", "santimetrekare",
        "kilometreküp", "santimetreküp",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    for w in ONES.iter().chain(TENS.iter().filter(|s| !s.is_empty())) {
        stems.push(w.to_string());
    }
    for (_, exp) in abbreviations() {
        if let Some(last) = exp.split_whitespace().last() {
            stems.push(last.to_string());
        }
    }
    for u in ["metre", "gram", "sisi", "mikrogram", "ünite", "kilokalori", "milisaniye", "desilitre"] {
        stems.push(u.to_string());
    }
    // sira eki kelimeleri (orn: birinci, ...): kisa liste yeterli degil; kural: sonu ...? 
    // Python: _ORDINAL_EXCEPTIONS.values() — asagida ekle
    for w in [
        "birinci", "ikinci", "üçüncü", "dördüncü", "beşinci", "altıncı",
        "yedinci", "sekizinci", "dokuzuncu", "onuncu", "yirminci",
        "otuzuncu", "kırkıncı", "ellinci", "altmışıncı", "yetmişinci",
        "sekseninci", "doksanıncı", "yüzüncü", "bininci", "milyonuncu",
        "milyarıncı",
    ] {
        stems.push(w.to_string());
    }
    stems.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    stems.dedup();
    let alts: Vec<String> = stems.iter().map(|s| regex_escape(s)).collect();
    let pat = format!(
        r"\b({}|\w+(?:inci|ıncı|üncü|uncu|nci|ncı|ncü|ncu))'([a-zçğıöşü]+)\b",
        alts.join("|")
    );
    re(&pat)
        .replace_all(t, |c: &Captures| {
            attach_suffix(
                c.get(1).map(|m| m.as_str()).unwrap_or(""),
                c.get(2).map(|m| m.as_str()).unwrap_or(""),
            )
        })
        .into_owned()
}

#[allow(dead_code)]
fn normalize_orthography(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let mut s: String = text.nfkc().collect();
    s = s
        .replace('\u{2019}', "'")
        .replace('\u{2018}', "'")
        .replace('`', "'")
        .replace('\u{00b4}', "'");
    s = s
        .replace('"', "")
        .replace('\u{201c}', "")
        .replace('\u{201d}', "")
        .replace('\u{201e}', "")
        .replace('\u{00ab}', "")
        .replace('\u{00bb}', "");
    s = s.replace("''", "");
    collapse_spaces(&s.trim().to_string())
}

fn lower_if_shouted(text: &str) -> String {
    let word_re = re(r"\b[^\W\d_]{2,}\b");
    let caps_re = re(r"\b[A-ZÇĞİÖŞÜQWX]{2,}\b");
    let mut words = Vec::new();
    let mut pos = 0;
    while pos <= text.len() {
        match word_re.find_from_pos(text, pos) {
            Ok(Some(m)) => {
                words.push(m.as_str().to_string());
                pos = m.end();
                if pos >= text.len() {
                    break;
                }
            }
            _ => break,
        }
    }
    if words.is_empty() {
        return text.to_string();
    }
    let capitals: Vec<String> = words
        .iter()
        .filter(|w| caps_re.is_match(w).unwrap_or(false))
        .cloned()
        .collect();
    if capitals.is_empty() {
        return text.to_string();
    }
    let spoken = capitals.iter().filter(|w| is_pronounceable(w)).count();
    if spoken >= 3 && (capitals.len() as f32) >= 0.7 * words.len() as f32 {
        turkish_lower(text)
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// Parcalama (split_model_text aynasi) + duraklama + on/ekler
// ---------------------------------------------------------------------------

pub const CHUNK_MAX_CHARS: usize = 300;
pub const CHUNK_FIRST_CHARS: usize = 80;
pub const SENTENCE_PAUSE: f32 = 0.12;
pub const CLAUSE_PAUSE: f32 = 0.06;

pub const PREFIX_TEXT: &str = "ee, ";
pub const SUFFIX_TEXT: &str = " ee.";
pub const PREFIX_INITIALS: &str = "öı";

/// Cumle sonu [.?!] + bosluklardan boler (lookbehindsiz esdeger).
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if ".?!".contains(chars[i])
            && i + 1 < chars.len()
            && chars[i + 1].is_whitespace()
        {
            out.push(chars[start..=i].iter().collect());
            // bosluklari atla
            i += 1;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            start = i;
            continue;
        }
        i += 1;
    }
    if start < chars.len() {
        out.push(chars[start..].iter().collect());
    }
    out.into_iter()
        .map(|s: String| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Cumlecik sonu [,;:] + bosluklardan boler.
fn split_clauses(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if ",;:".contains(chars[i])
            && i + 1 < chars.len()
            && chars[i + 1].is_whitespace()
        {
            out.push(chars[start..=i].iter().collect());
            i += 1;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            start = i;
            continue;
        }
        i += 1;
    }
    if start < chars.len() {
        out.push(chars[start..].iter().collect());
    }
    out
}

fn pack(pieces: &[String], max_chars: usize) -> Vec<String> {
    fn pack_inner(pieces: &[String], max_chars: usize, out: &mut Vec<String>) {
        let mut cur = String::new();
        for p in pieces {
            let n = p.chars().count();
            if n > max_chars {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                if p.contains(' ') {
                    // kelimelere bolup ozyinele
                    let words: Vec<String> =
                        p.split(' ').map(|w| w.to_string()).collect();
                    pack_inner(&words, max_chars, out);
                } else {
                    // sert kesim (harf sayisina gore)
                    let chars: Vec<char> = p.chars().collect();
                    for block in chars.chunks(max_chars) {
                        out.push(block.iter().collect());
                    }
                }
                continue;
            }
            if !cur.is_empty() && cur.chars().count() + 1 + n > max_chars {
                out.push(std::mem::take(&mut cur));
                cur = p.clone();
            } else if cur.is_empty() {
                cur = p.clone();
            } else {
                cur = format!("{cur} {p}");
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    let mut out = Vec::new();
    pack_inner(pieces, max_chars, &mut out);
    out
}

/// Konusulur metin -> sentezlenecek parcalar (say/stream ile ayni).
pub fn split_model_text(text: &str) -> Vec<String> {
    split_model_text_full(text, CHUNK_MAX_CHARS, Some(CHUNK_FIRST_CHARS))
}

fn split_model_text_full(
    text: &str,
    max_chars: usize,
    first_chars: Option<usize>,
) -> Vec<String> {
    let sentences: Vec<String> = split_sentences(text.trim());
    let mut pieces: Vec<String> = Vec::new();
    for s in &sentences {
        if s.chars().count() > max_chars {
            // Once cumleciklere bol
            let clauses: Vec<String> = split_clauses(s)
                .into_iter()
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect();
            pieces.extend(pack(&clauses, max_chars));
        } else {
            pieces.push(s.clone());
        }
    }
    if pieces.is_empty() {
        return Vec::new();
    }
    let chunks = match first_chars {
        Some(fc) => {
            let mut head = String::new();
            let mut used = 0usize;
            for p in &pieces {
                if !head.is_empty()
                    && head.chars().count() + 1 + p.chars().count() > fc
                {
                    break;
                }
                head = if head.is_empty() {
                    p.clone()
                } else {
                    format!("{head} {p}")
                };
                used += 1;
            }
            let mut chunks = vec![head];
            chunks.extend(pack(&pieces[used..], max_chars));
            chunks
        }
        None => pack(&pieces, max_chars),
    };
    let letter = |s: &str| s.chars().any(|c| c.is_alphabetic());
    chunks.into_iter().filter(|c| letter(c)).collect()
}

/// Parca sonu duraklamasi: cumle sonu 0.12 sn, cumlecik 0.06 sn.
pub fn chunk_pause(chunk: &str) -> f32 {
    match chunk.trim_end().chars().last() {
        Some('.') | Some('?') | Some('!') => SENTENCE_PAUSE,
        _ => CLAUSE_PAUSE,
    }
}

/// Parca basi prefix'i (ilk harf ö/ı ise "ee, ", yoksa "").
pub fn chunk_prefix(chunk: &str) -> &'static str {
    match chunk.chars().next() {
        Some(c) if PREFIX_INITIALS.contains(c) => PREFIX_TEXT,
        _ => "",
    }
}

pub const CHUNK_SUFFIX: &str = SUFFIX_TEXT;

#[cfg(test)]
mod tests {
    use super::*;

    fn check_cases(path: &str) {
        let data = std::fs::read_to_string(path).expect("batarya dosyasi");
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(&data).expect("batarya json");
        let mut fail = 0;
        for c in &cases {
            let input = c["in"].as_str().unwrap_or("");
            let expected = c["out"].as_str().unwrap_or("");
            let got = normalize_for_model_text(input);
            if got != expected {
                fail += 1;
                println!("FARK\n  in:  {input:?}\n  bek: {expected:?}\n  bul: {got:?}");
            }
        }
        assert_eq!(fail, 0, "{fail} batarya vakasi tutmadi");
    }

    #[test]
    fn normalizer_battery_matches_python() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models/antalia_ref/normalizer_cases.json");
        check_cases(path.to_str().unwrap());
    }

    #[test]
    fn chunker_first_sentence_first() {
        // Ilk parca ilk cumleyi icerir (akiskan baslangic).
        let text = "Merhaba. Size nasıl yardımcı olabilirim? Bugün hava çok güzel ve kuşlar ötüyor.";
        let chunks = split_model_text(text);
        assert!(!chunks.is_empty());
        assert!(chunks[0].starts_with("Merhaba."));
    }

    #[test]
    fn chunker_long_sentence_splits() {
        let long = format!("{} {}", "kelime ".repeat(120), "bitti.");
        let chunks = split_model_text(&long);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= CHUNK_MAX_CHARS + 20);
        }
    }
}
