//! Map and player NAMES: the two layers of encoding between a title's bytes
//! and the title a person reads.
//!
//! This lives in `gbx`, beside the container, for the reason the crate exists:
//! it was written twice in one afternoon otherwise — once in `tmmaps header`
//! reading a `.Map.Gbx`, once in `tmsite names` reading trackmania.io — and
//! two decoders of one encoding is how a name comes out right in one tool and
//! wrong in the other with nothing failing.
//!
//! ## Layer 1 — ManiaPlanet text markup (`strip_fmt`)
//!
//! A map name is stored WITH its decoration in it. 126859's is literally
//!
//! ```text
//! $o$i$aa0Kack$05ay Re$09alo$6a0ad$aa0ed $4f0#290
//! ```
//!
//! which is "Kacky Reloaded #290" wearing three colours, a bold and an italic.
//! Both the map file and trackmania.io hand it over in that form, so a
//! comparison of raw strings reports a difference on every decorated map and
//! says nothing about whether the NAME differs.
//!
//! ## Layer 2 — XML attribute escapes (`unescape_xml`)
//!
//! The `.Map.Gbx` header carries its fields as an XML chunk, so a name with an
//! apostrophe or an ampersand arrives escaped: 208024's header holds
//! `Miru&apos;s Hell 2` and 285268's `Pain ft Mango &amp; Teuflum`. Reading the
//! attribute without unescaping makes both look like names we got wrong, which
//! is exactly the false positive that would bury the real ones.
//!
//! The two layers compose in that order: unescape the XML, then strip the
//! markup.

/// Strip ManiaPlanet text formatting.
///
/// * `$$` — a literal `$`.
/// * `$` + three hex digits — an RGB colour.
/// * `$g $i $m $n $o $s $t $w $z $< $>` — style toggles and resets.
/// * `$h $l $p` — link markers, optionally `[target]`; the visible text
///   between them is kept, the marker and its target are not.
///
/// `$t` (uppercase) is dropped rather than applied: applying it would produce a
/// string that appears in no file, and this function removes decoration, it
/// does not render.
pub fn strip_fmt(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] != '$' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        // A trailing lone `$` is not an escape.
        let Some(&c) = b.get(i + 1) else {
            out.push('$');
            break;
        };
        if c == '$' {
            out.push('$');
            i += 2;
            continue;
        }
        if i + 3 < b.len() && b[i + 1..i + 4].iter().all(|c| c.is_ascii_hexdigit()) {
            i += 4;
            continue;
        }
        match c.to_ascii_lowercase() {
            'h' | 'l' | 'p' => {
                i += 2;
                if b.get(i) == Some(&'[') {
                    match b[i..].iter().position(|&c| c == ']') {
                        Some(e) => i += e + 1,
                        None => i = b.len(),
                    }
                }
            }
            'g' | 'i' | 'm' | 'n' | 'o' | 's' | 't' | 'w' | 'z' | '<' | '>' => i += 2,
            // Not an escape this renderer knows: keep both characters rather
            // than silently eating a real one.
            _ => {
                out.push('$');
                i += 1;
            }
        }
    }
    out
}

/// Undo XML attribute escaping: the five predefined entities, plus numeric
/// character references in decimal and hex.
///
/// An entity this does not recognise is left exactly as written — a name that
/// really contains the text `&foo;` must survive the round trip, and inventing
/// a character for it would be worse than printing it.
pub fn unescape_xml(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let b: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] != '&' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        // An entity is `&…;` within a short window; anything longer is a
        // literal ampersand in the text.
        let Some(semi) = b[i..].iter().take(12).position(|&c| c == ';') else {
            out.push('&');
            i += 1;
            continue;
        };
        let body: String = b[i + 1..i + semi].iter().collect();
        let rep = match body.as_str() {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => {
                let n = if let Some(h) = body.strip_prefix("#x").or_else(|| body.strip_prefix("#X")) {
                    u32::from_str_radix(h, 16).ok()
                } else {
                    body.strip_prefix('#').and_then(|d| d.parse::<u32>().ok())
                };
                n.and_then(char::from_u32)
            }
        };
        match rep {
            Some(c) => {
                out.push(c);
                i += semi + 1;
            }
            None => {
                out.push('&');
                i += 1;
            }
        }
    }
    out
}

/// A name as a reader sees it: XML-unescaped, then stripped of markup.
pub fn plain(s: &str) -> String {
    strip_fmt(&unescape_xml(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_real_corpus_names() {
        // Every input here is a byte-for-byte string out of a map in this
        // project's corpus or out of trackmania.io's response for it.
        assert_eq!(strip_fmt("$o$i$aa0Kack$05ay Re$09alo$6a0ad$aa0ed $4f0#290"), "Kacky Reloaded #290");
        assert_eq!(strip_fmt("$903Welcome☺$903to $903wiggles"), "Welcome☺to wiggles");
        assert_eq!(
            strip_fmt("$60CM$71Di$82Dr$93Eu$A4E'$B5Fs$C6F $C6FH$D5Fe$D4Fl$E2Fl 2"),
            "Miru's Hell 2"
        );
        assert_eq!(strip_fmt("[object Object]"), "[object Object]");
    }

    #[test]
    fn keeps_what_is_not_an_escape() {
        assert_eq!(strip_fmt("100$$"), "100$");
        assert_eq!(strip_fmt("cost $5"), "cost $5");
        assert_eq!(strip_fmt("$h[www]click$h"), "click");
        assert_eq!(strip_fmt("trailing$"), "trailing$");
    }

    #[test]
    fn unescapes_xml_attributes() {
        assert_eq!(unescape_xml("Miru&apos;s Hell 2"), "Miru's Hell 2");
        assert_eq!(unescape_xml("Pain ft Mango &amp; Teuflum"), "Pain ft Mango & Teuflum");
        assert_eq!(unescape_xml("&#65;&#x42;"), "AB");
        // Not an entity: left alone rather than guessed at.
        assert_eq!(unescape_xml("Q&A"), "Q&A");
        assert_eq!(unescape_xml("&notanentity;"), "&notanentity;");
    }

    #[test]
    fn the_two_layers_compose() {
        assert_eq!(plain("$63fP$fffain ft Mango &amp; Teuflum"), "Pain ft Mango & Teuflum");
    }

    /// THE CONTROL for the pair above: a decorated, escaped name must come out
    /// DIFFERENT from its raw form. Without this, a `plain()` that did nothing
    /// at all would pass every "is the name right" check in the audit on the
    /// undecorated majority of the corpus and quietly fail on the rest.
    #[test]
    fn plain_actually_changes_a_decorated_name() {
        let raw = "$o$i$aa0Kack$05ay Re$09alo$6a0ad$aa0ed $4f0#290";
        assert_ne!(plain(raw), raw);
        let raw = "Miru&apos;s Hell 2";
        assert_ne!(plain(raw), raw);
    }
}
