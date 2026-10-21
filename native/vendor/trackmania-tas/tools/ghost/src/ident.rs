//! Identity: the car skin, the display name, the 3-letter trigram, and the
//! foreign identifiers that ride along with a borrowed container.
//!
//! OUR RUN CAN BE COMPLETELY OURS AND THE FILE STILL SOMEBODY ELSE'S. A
//! synthesised tape is built on a carrier and inherits that carrier's body:
//! its login, its club tag, its zone, its personal skin, the storage-object
//! uuid inside the skin path, and the server-reported account id. A strip-list
//! aimed at one of those leaves the others, which is how five published clips
//! had to be withdrawn in one night.
//!
//! So this module does not have a strip-list. It CLASSIFIES every identity
//! string in the container, prints them all with their offsets, and
//! `--anonymise` clears every class at once -- and `ghost verify` fails a file
//! that still carries any of them.

use gbx::container::{body_strings_in, replace_strings, write_gbx, Container};
use crate::cli::{die, flag, has};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Skin,
    Locator,
    Zone,
    Nickname,
    Trigram,
    ClubTag,
    Login,
    AccountId,
    MapUid,
    /// The carrier player's ranked-badge state, e.g.
    /// `Prestige=Yes&Level=1&Year=2026&Mode=Ranked&Medal=Master&SubRank=3`.
    /// A per-PLAYER field, sitting in the head of chunk 0x03092000 with the
    /// skins and the display name. Found 2026-08-22 by raw-stringing five
    /// published ghosts: 16 published files on 5 maps carry a stranger's
    /// badge, and it survived every anonymiser because nobody had listed it.
    Prestige,
    Other,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::Skin => "skin",
            Role::Locator => "locator URL",
            Role::Zone => "zone",
            Role::Nickname => "display name",
            Role::Trigram => "trigram",
            Role::ClubTag => "club tag",
            Role::Login => "login",
            Role::AccountId => "account id",
            Role::MapUid => "map uid",
            Role::Prestige => "ranked badge",
            Role::Other => "",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub role: Role,
    pub at: usize,
    pub len: usize,
    pub s: String,
}


/// Classify the identity strings a ghost carries.
///
/// This is STRUCTURAL, not a string search. `CGameCtnGhost`'s main chunk
/// (`0x03092000`) holds, in order: the player model, the skin pack descriptors
/// (each a checksum, a path and a locator URL), the display name, then the
/// compressed record -- and AFTER the record, the trigram, the zone and the
/// club tag. The account id is its own chunk `0x0309200F` and the map uid is
/// `0x03092010`.
///
/// Anything in the chunk this does not recognise is still returned as `Other`
/// and still printed: the module never hides a string it does not understand,
/// because the last five identity bugs were all about a field nobody listed.
pub fn scan(c: &Container) -> Vec<Field> {
    let body = c.body();
    let mut out: Vec<Field> = Vec::new();
    let chunks = c.chunks();

    // the main ghost chunk, and where the record blob starts inside it
    let main = chunks.iter().find(|k| k.0 == 0x03092000);
    let rec = gbx::recwrite::find_rec_site(body).ok().map(|s| s.hdr);
    // A REPLAY nests its ghost node without a skippable 0x03092000 chunk, so
    // the structural walk has nothing to anchor on. Fall back to the whole
    // region in front of the input chunk, which is where the same fields sit.
    let fallback_span = || -> (usize, usize) {
        // Start AFTER the embedded map. A replay carries a whole map, and that
        // map has its own author login and zone in it -- scanning across it
        // finds the MAP AUTHOR and offers to rename them, which is a different
        // person and a different file.
        let start = c.embedded_map().map(|(o, n)| o + n).unwrap_or(0);
        let end = gbx::tape::find_inputs_chunk(body)
            .map(|x| x.0)
            .unwrap_or(body.len())
            .max(start);
        (start, end)
    };
    let (mpoff, mpend) = match main {
        Some(&(_, _, poff, sz)) => (poff, poff + sz),
        None => fallback_span(),
    };
    {
        let split = rec.filter(|r| *r > mpoff && *r < mpend).unwrap_or(mpend);
        // --- before the record: model, skins, display name
        let head = body_strings_in(body, mpoff, split);
        for (i, b) in head.iter().enumerate() {
            let role = if b.s.contains("Skins\\Models\\") || b.s.contains("Skins/Models/") {
                Role::Skin
            } else if b.s.starts_with("http://") || b.s.starts_with("https://") {
                Role::Locator
            } else if b.s.starts_with("Prestige=") {
                Role::Prestige
            } else if i + 1 == head.len() {
                // the string immediately before the record is the ghost's
                // displayed player name
                Role::Nickname
            } else {
                Role::Other
            };
            out.push(Field { role, at: b.at, len: b.len, s: b.s.clone() });
        }
        // --- after the record: trigram, zone, club tag
        let tailstart = rec
            .and_then(|r| {
                gbx::recwrite::find_rec_site(body).ok().map(|s| r + 12 + s.csize)
            })
            .unwrap_or(mpoff);
        let tail = body_strings_in(body, tailstart, mpend);
        let zi = tail.iter().position(|b| b.s.starts_with("World|"));
        for (i, b) in tail.iter().enumerate() {
            let role = match zi {
                Some(z) if i == z => Role::Zone,
                Some(z) if i + 1 == z => Role::Trigram,
                Some(z) if i == z + 1 => Role::ClubTag,
                _ => Role::Other,
            };
            out.push(Field { role, at: b.at, len: b.len, s: b.s.clone() });
        }
    }
    // --- the REPLAY's own author node, body chunk 0x03093018.
    //
    // A `.Replay.Gbx` carries a fourth copy of the driver, past the nested
    // ghost node entirely:
    //
    //     lookback titleId ("TMStadium")
    //     u32      authorVersion
    //     str      login, str nickname, str zone, str extra
    //
    // The walk above anchors on the ghost node and stops before this, so the
    // block was never seen: an `--anonymise` that reported success left the
    // driver's login and nickname here, and V3 could not see them either. The
    // offsets are read, not assumed -- the first guess (`poff + 8`) was wrong
    // by the whole title-id lookback and silently found nothing, which is what
    // a wrong offset always looks like.
    if let Some(&(_, _, poff, sz)) = chunks.iter().find(|k| k.0 == 0x03093018) {
        let end = poff + sz;
        let rd_u32 = |o: usize| -> Option<u32> {
            body.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let mut o = poff;
        // the title id, as a lookback string
        if rd_u32(o) == Some(0x4000_0000) {
            o += 4;
            let n = rd_u32(o).unwrap_or(0) as usize;
            o += 4 + n;
        } else {
            o += 4;
        }
        o += 4; // authorVersion
        for role in [Role::Login, Role::Nickname, Role::Zone] {
            let Some(n) = rd_u32(o).map(|v| v as usize) else { break };
            if n > 256 || o + 4 + n > end {
                break;
            }
            if let Ok(s) = std::str::from_utf8(&body[o + 4..o + 4 + n]) {
                out.push(Field { role, at: o, len: n, s: s.to_string() });
            }
            o += 4 + n;
        }
    }
    // --- the SECOND ranked badge, skippable chunk `0x0309202E`.
    //
    // The badge is written twice: once as a bare string in the skin block of
    // `0x03092000`, and once as a chunk of its own -- `str` then one byte --
    // which the walk above never reaches because it stops at `0x03092000`'s
    // end. Stripping only the first left the carrier player's season standing
    // in the file and `ghost identity show` reporting a clean pass, which is
    // the same shape as every other identity bug here: a site nobody listed.
    // Found by counting the raw string in the bytes AFTER an anonymise pass
    // said it had cleared it -- 2 occurrences before, 1 after.
    if let Some(&(_, _, poff, sz)) = chunks.iter().find(|k| k.0 == 0x0309202E) {
        if sz >= 4 {
            let n = u32::from_le_bytes(body[poff..poff + 4].try_into().unwrap()) as usize;
            if n > 0 && n <= sz.saturating_sub(4) {
                if let Ok(s) = std::str::from_utf8(&body[poff + 4..poff + 4 + n]) {
                    out.push(Field {
                        role: Role::Prestige,
                        at: poff,
                        len: n,
                        s: s.to_string(),
                    });
                }
            }
        }
    }
    // --- the account id and the map uid.
    //
    // These are NOT skippable chunks: `0x0309200F` and `0x03092010` are written
    // inline, chunk id then data, with no `PIKS` marker and no size -- so a
    // scan that only knows skippable chunks walks straight past both, which is
    // how an account id survives an anonymiser. They are found by their id
    // bytes and then validated by parsing what follows.
    if let Some(f) = inline_string_chunk(body, 0x0309200F, Role::AccountId) {
        out.push(f);
    }
    if let Some(f) = inline_string_chunk(body, 0x03092010, Role::MapUid) {
        out.push(f);
    }
    out.sort_by_key(|f| f.at);
    // The inline-chunk lookup and the string walk can both land on the same
    // string; keep the named one.
    out.dedup_by(|b, a| {
        if a.at == b.at {
            if a.role == Role::Other {
                a.role = b.role;
            }
            true
        } else {
            false
        }
    });
    out
}

/// Find an inline (non-skippable) chunk that holds one length-prefixed string.
/// The chunk id may be followed by a `0x40000000` lookback marker.
fn inline_string_chunk(body: &[u8], cid: u32, role: Role) -> Option<Field> {
    let idb = cid.to_le_bytes();
    let mut i = 0usize;
    while i + 12 <= body.len() {
        if body[i..i + 4] == idb {
            for skip in [4usize, 8] {
                let p = i + skip;
                if p + 4 > body.len() {
                    continue;
                }
                if skip == 8 && u32::from_le_bytes(body[i + 4..i + 8].try_into().unwrap()) != 0x4000_0000 {
                    continue;
                }
                let n = u32::from_le_bytes(body[p..p + 4].try_into().unwrap()) as usize;
                if n == 0 || n > 64 || p + 4 + n > body.len() {
                    continue;
                }
                if let Ok(s) = std::str::from_utf8(&body[p + 4..p + 4 + n]) {
                    if s.chars().all(|c| c.is_ascii_graphic()) {
                        return Some(Field { role, at: p, len: n, s: s.to_string() });
                    }
                }
            }
        }
        i += 1;
    }
    None
}

pub fn print(c: &Container) {
    let f = scan(c);
    let named: Vec<&Field> = f.iter().filter(|x| x.role != Role::Other && x.role != Role::MapUid).collect();
    if named.is_empty() {
        println!("identity      (no identity strings found)");
        return;
    }
    println!("identity");
    for x in named {
        println!("  {:<12} @{:<7} {:?}", x.role.label(), x.at, x.s);
    }
}

pub const DEFAULT_SKIN: &str = "Skins\\Models\\CarSport\\TAS.zip";

pub fn cmd(a: &[String]) {
    let what = a.first().map(|s| s.as_str()).unwrap_or_else(|| die("ghost identity <show|set>"));
    let rest = &a[1..];
    match what {
        "show" => {
            let c = Container::load(&rest[0]).unwrap_or_else(|e| die(e));
            let f = scan(&c);
            println!("{:<14} {:>8}  {}", "role", "offset", "value");
            for x in &f {
                if x.role == Role::Other && x.s.len() < 3 {
                    continue;
                }
                println!("{:<14} {:>8}  {:?}", x.role.label(), x.at, x.s);
                // a skin path is preceded by its PackDesc checksum (sha256 of the zip,
                // measured 2026-09-12): print it, it is half of what makes a skin
                // fetchable and a zero one makes the game try to "update" the file
                if x.role == Role::Skin && x.at >= 32 {
                    let h: String = c.body()[x.at - 32..x.at].iter().map(|b| format!("{b:02x}")).collect();
                    let zero = h.chars().all(|ch| ch == '0');
                    println!("{:<14} {:>8}  {}{}", "  checksum", x.at - 32, h, if zero { "  (ZERO -- the game will try to update this file from its locator)" } else { "" });
                }
            }
        }
        "set" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost identity set IN OUT [flags]"));
            let out = rest.get(1).unwrap_or_else(|| die("ghost identity set IN OUT [flags]"));
            let c = Container::load(inp).unwrap_or_else(|e| die(e));
            let fields = scan(&c);
            let anon = has(rest, "--anonymise") || has(rest, "--anonymize");
            let name = flag(rest, "--name");
            let trigram = flag(rest, "--trigram");
            let skin = flag(rest, "--skin");
            // --locator URL and --skin-file ZIP | --checksum HEX64 (2026-09-13): the skin
            // PackDesc is `u8 3 | sha256(zip) | path | locator`; a ghost that names a zip
            // the player lacks needs the real hash and the URL for the game to fetch it.
            let locator = flag(rest, "--locator");
            if let Some(u) = locator {
                if !(u.is_empty() || u.starts_with("http://") || u.starts_with("https://")) {
                    die(format!("--locator {u:?}: not an http(s) URL"));
                }
            }
            let checksum: Option<[u8; 32]> = match (flag(rest, "--skin-file"), flag(rest, "--checksum")) {
                (Some(f), _) => Some(gbx::sha::sha256_file(std::path::Path::new(f)).unwrap_or_else(|e| die(format!("--skin-file {f}: {e}")))),
                (None, Some(h)) => {
                    let h = h.trim();
                    if h.len() != 64 { die("--checksum wants 64 hex chars (sha256 of the zip)"); }
                    let mut o = [0u8; 32];
                    for i in 0..32 { o[i] = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).unwrap_or_else(|_| die("--checksum: not hex")); }
                    Some(o)
                }
                _ => None,
            };
            let login = flag(rest, "--login");
            let zone = flag(rest, "--zone");
            let clubtag = flag(rest, "--clubtag");
            if let Some(t) = trigram {
                if t.len() != 3 {
                    die(format!("a trigram is exactly 3 characters ({:?} is {})", t, t.len()));
                }
            }
            let mut edits: Vec<(usize, usize, Vec<u8>)> = Vec::new();
            // Padding an unresizable field to `xxxx…` is a fallback, not the
            // answer: the 126 clean containers in this repo carry NO account
            // id, and a file with 22 `x`s is a third category nobody has seen.
            // So when a dedicated server is available, SHORTEN and let the
            // plain oracle adjudicate -- the control that already runs below
            // deletes the output if the edit changed what the file does.
            // `--pad-ids` forces the old behaviour.
            let have_server = !has(rest, "--no-oracle")
                && crate::oracle::server_dir(flag(rest, "--server"))
                    .join("TrackmaniaServer")
                    .exists();
            let pad_ids = has(rest, "--pad-ids") || !have_server;
            let mut zero_cksum: Vec<usize> = Vec::new();
            let mut log: Vec<String> = Vec::new();
            // Which edits could be resized safely, so an anonymisation that
            // cannot shorten a string can pad instead of giving up. In a ghost
            // the identity sits inside skippable chunk 0x03092000 and shortening
            // is free; in a REPLAY the same strings sit in a nested node whose
            // offsets something else depends on, and shortening the driver's
            // name there produces a file that reads back perfectly and
            // validates to nothing.
            let chunks_ok = |at: usize, old: usize| -> bool {
                c.chunks()
                    .iter()
                    .filter(|(_, coff, poff, sz)| {
                        c.embedded_map().map_or(true, |(mo, ms)| *coff >= mo + ms || poff + sz <= mo)
                    })
                    .any(|(_, _, poff, sz)| at >= *poff && at + 4 + old <= poff + sz)
            };
            for f in &fields {
                let newv: Option<String> = match f.role {
                    Role::Skin => skin
                        .map(|s| if s == "default" { DEFAULT_SKIN.to_string() } else { s.to_string() })
                        .or(if anon { Some(DEFAULT_SKIN.to_string()) } else { None }),
                    Role::Nickname => name.map(|s| s.to_string()).or(if anon { Some("TAS".into()) } else { None }),
                    Role::Trigram => trigram.map(|s| s.to_string()).or(if anon { Some("TAS".into()) } else { None }),
                    Role::Login => login.map(|s| s.to_string()).or(if anon { Some("TAS".into()) } else { None }),
                    // A ZONE IS A PERSON'S COUNTRY, and 21 published ghosts
                    // carry a stranger's -- Austria on nine 165922 files,
                    // Russia on the Leto author-cuts, the United Kingdom on the
                    // Blev carrier -- against 137 that carry none. It is the
                    // same shape as the account id and it was on nobody's
                    // strip-list.
                    //
                    // IT IS STILL NOT BLANKED, AND THE SUITE IS WHY. The zone
                    // is the ANCHOR this scanner finds the trigram and the club
                    // tag by: `World|...` is the only self-identifying string
                    // in that block, so the trigram is "the string before it"
                    // and the club tag "the string after". Emptying it makes
                    // both unfindable -- O7 caught it immediately, asking for
                    // trigram VJX and reading back None. Blanking a field by
                    // destroying the landmark that locates its neighbours would
                    // trade a named leak for two silent ones.
                    //
                    // So it is set only when asked (`--zone ""` works), and
                    // `ghost verify` V3 reports a carried zone. Doing it
                    // properly needs the scanner to find the trigram and club
                    // tag structurally rather than relative to the zone, which
                    // is a change to make with the corpus in front of you.
                    Role::Zone => zone.map(|s| s.to_string()),
                    Role::ClubTag => clubtag.map(|s| s.to_string()).or(if anon { Some(String::new()) } else { None }),
                    Role::Locator => locator.map(|s| s.to_string()).or(if anon { Some(String::new()) } else { None }),
                    Role::AccountId => if anon { Some(String::new()) } else { None },
                    Role::Prestige => if anon { Some(String::new()) } else { None },
                    _ => None,
                };
                if let Some(v) = newv {
                    let mut v = v;
                    // An anonymisation must not fail just because it cannot
                    // shorten: pad to the original byte length instead. `x`
                    // repeated is not a plausible account id or URL, which is
                    // the point.
                    // PAD ONLY WHEN NOTHING CAN PROVE THE SHRINK.
                    //
                    // A body is parsed serially, so a top-level INLINE string
                    // -- which is exactly where a ghost keeps its account id,
                    // chunk 0x0309200F, no size word anywhere -- can shrink
                    // safely. `chunks_ok` only knows skippable chunks, so it
                    // said no and the anonymiser padded a stranger's account id
                    // to twenty-two `x`s instead of removing it. That still
                    // reads as AN ACCOUNT ID: `ghost verify` V3 and the
                    // integrity gate's C-ident both refuse it, and the corpus
                    // rule is 152 files with login TAS and NO id, none in
                    // between. The oracle is what proves the shrink, and the
                    // no-op control below already runs it.
                    if v.len() != f.len && !chunks_ok(f.at, f.len) && anon && v.len() < f.len && pad_ids
                    {
                        let pad = "x".repeat(f.len);
                        log.push(format!(
                            "  {:<12} padded to {} bytes -- it cannot be resized in this container \
                             and no oracle was available to prove a shrink (pass --map and a \
                             server to remove it outright)",
                            f.role.label(),
                            f.len
                        ));
                        v = pad;
                    }
                    if v != f.s {
                        log.push(format!("  {:<12} {:?} -> {:?}", f.role.label(), f.s, v));
                        if f.role == Role::Skin {
                            zero_cksum.push(f.at);
                        }
                        edits.push((f.at, f.len, v.into_bytes()));
                    }
                }
            }
            // a locator on a ghost whose locator string is EMPTY: the scanner never lists
            // an empty string, so it is addressed structurally -- the length word right
            // after the skin path (as `ghost static` does)
            if let Some(u) = locator {
                if !fields.iter().any(|f| f.role == Role::Locator) {
                    if let Some(sk) = fields.iter().find(|f| f.role == Role::Skin) {
                        let at = sk.at + 4 + sk.len;
                        let old_len = u32::from_le_bytes(c.body()[at..at + 4].try_into().unwrap()) as usize;
                        if old_len > 1024 { die(format!("the word after the skin path reads {old_len}, not a plausible locator length")); }
                        log.push(format!("  {:<12} (empty) -> {:?}", "locator", u));
                        edits.push((at, old_len, u.as_bytes().to_vec()));
                    }
                }
            }
            if edits.is_empty() && checksum.is_none() {
                // A NO-OP IS A RESULT, AND IT WAS KILLING THE CALLER.
                //
                // `die` here is right for a person at a terminal: they asked
                // for a change and there is none to make. It is wrong for
                // `ghost regen`, whose finishing pass calls this IN PROCESS to
                // anonymise what it just wrote -- on a file that is already
                // anonymous (every re-regeneration of a published ghost is)
                // this exited the whole command, silently, between "the gate
                // passed" and every acceptance check that comes after it: the
                // donor-byte provenance, the channel liveness, the carrier
                // read-back. The regeneration then LEFT A FILE BEHIND that
                // nothing had checked, and on 227654 that file was a decoy the
                // checks would have refused. Measured 2026-08-24; the log's
                // tell is that it ends at "the finishing pass".
                if has(rest, "--allow-noop") {
                    println!("  identity: nothing to change -- this file is already anonymous");
                    std::fs::copy(inp, out)
                        .unwrap_or_else(|e| die(format!("copying {inp} to {out}: {e}")));
                    return;
                }
                die("nothing to change (`ghost identity show` lists what is there)");
            }
            edits.sort_by_key(|e| e.0);
            let mut pre = c.body().to_vec();
            for at in &zero_cksum {
                if *at >= 32 {
                    for b in pre[at - 32..*at].iter_mut() {
                        *b = 0;
                    }
                }
            }
            if let Some(h) = checksum {
                for f in fields.iter().filter(|f| f.role == Role::Skin) {
                    if f.at >= 32 {
                        pre[f.at - 32..f.at].copy_from_slice(&h);
                        log.push(format!("  {:<12} checksum -> sha256 {}", "skin", h.iter().map(|b| format!("{b:02x}")).collect::<String>()));
                    }
                }
            }
            let protect = c.embedded_map();
            let body = replace_strings(&pre, &edits, protect).unwrap_or_else(|e| die(e));
            let unframed = gbx::container::unframed_edits();
            // THE HEADER IS A SECOND CONTAINER. A .Replay.Gbx keeps the
            // driver's login, nickname, zone and account id in its header
            // user-data as well, and nothing here read them: an --anonymise
            // that reported success left `GothMommyTM` and his account id in
            // the header of a file this project then published. The map's own
            // attribution in the same header is left alone -- see `hdr`.
            let mut gbx = c.gbx.clone();
            let mut hdr_log: Vec<String> = Vec::new();
            if anon {
                if let Some(e) = crate::hdr::rewrite(&c, true, name, None) {
                    gbx.user_data = e.user_data;
                    hdr_log = e.log;
                }
            }
            write_gbx(&gbx, body, out).unwrap_or_else(|e| die(e));
            // control: read it back and require every field to be what we asked
            let c2 = Container::load(out).unwrap_or_else(|e| die(e));
            let after = scan(&c2);
            for (role, want) in [
                (Role::Nickname, name),
                (Role::Trigram, trigram),
                (Role::Login, login),
            (Role::ClubTag, clubtag),
            ] {
                if let Some(w) = want {
                    let got = after.iter().find(|f| f.role == role).map(|f| f.s.clone());
                    if got.as_deref() != Some(w) {
                        die(format!(
                            "read-back control FAILED: {} is {:?}, asked for {:?}",
                            role.label(),
                            got,
                            w
                        ));
                    }
                }
            }
            // an anonymise that leaves an identifier behind is the exact failure
            // this exists to prevent, so it is checked rather than assumed
            if anon {
                let left: Vec<String> = after
                    .iter()
                    .filter(|f| {
                        matches!(f.role, Role::AccountId | Role::Locator | Role::Prestige)
                            && !f.s.is_empty()
                            && !f.s.chars().all(|c| c == 'x')
                    })
                    .map(|f| format!("{} {:?}", f.role.label(), f.s))
                    .collect();
                if !left.is_empty() {
                    die(format!("--anonymise left identifiers behind: {:?}", left));
                }
            }
            // and the same question of the header, which is where this check
            // was blind: `--anonymise` used to leave the driver's login and
            // nickname sitting in the header of a file it had just reported
            // clean.
            if anon {
                let left: Vec<String> = crate::hdr::header_driver_identity(&c2)
                    .into_iter()
                    .filter(|(_, v)| {
                        !v.is_empty()
                            && v != name.unwrap_or("TAS")
                            && !v.chars().all(|c| c == 'x')
                    })
                    .map(|(w, v)| format!("{} {:?}", w, v))
                    .collect();
                if !left.is_empty() {
                    die(format!(
                        "--anonymise left DRIVER identity in the HEADER: {:?}",
                        left
                    ));
                }
            }
            println!("wrote {}", out);
            for l in log {
                println!("{}", l);
            }
            for l in hdr_log {
                println!("{}", l);
            }
            println!("  read-back control OK");
            // THE CONTROL THAT MATTERS: a cosmetic edit must not change the
            // physics. Nothing in a read-back can see a file that still reads
            // correctly and no longer loads, so ask the plain oracle the same
            // question before and after.
            if !has(rest, "--no-oracle") {
                let server = crate::oracle::server_dir(flag(rest, "--server"));
                if server.join("TrackmaniaServer").exists() {
                    let mode = match flag(rest, "--map") {
                        Some(m) => crate::oracle::MapsMode::One(std::path::Path::new(m)),
                        None => crate::oracle::MapsMode::Empty,
                    };
                    let before = crate::oracle::validate(&server, std::path::Path::new(inp), mode, "id-a");
                    let after = crate::oracle::validate(&server, std::path::Path::new(out), mode, "id-b");
                    match (before, after) {
                        (Ok(b), Ok(a2)) => {
                            if b.time_ms.is_none() && a2.time_ms.is_none() {
                                // A CONTROL THAT CANNOT FAIL IS NOT A CONTROL.
                                // With no map staged the server DNFs both files
                                // and the equal comparison prints OK -- which is
                                // how an edit that broke a file would read as
                                // proved. Say what happened instead.
                                println!(
                                    "  oracle no-op control VACUOUS: the file DNFs both before and \
                                     after, so this proves nothing about the edit. Pass --map \
                                     MAP.Map.Gbx (a pure ghost needs one) to make it a control."
                                );
                                if !unframed.is_empty() {
                                    let _ = std::fs::remove_file(out);
                                    die(format!(
                                        "REFUSED and deleted {}: {} edit(s) changed a string's \
                                         length with no enclosing chunk to correct ({}), and the \
                                         oracle control was vacuous, so nothing has checked that \
                                         the file still simulates.",
                                        out,
                                        unframed.len(),
                                        unframed.join(", ")
                                    ));
                                }
                            } else if b.time_ms == a2.time_ms {
                                println!("  oracle no-op control OK: {} before and after", b.secs());
                            } else {
                                let _ = std::fs::remove_file(out);
                                die(format!(
                                    "ORACLE NO-OP CONTROL FAILED: {} before, {} after. A cosmetic \
                                     edit changed what the file does. {} has been DELETED.",
                                    b.secs(),
                                    a2.secs(),
                                    out
                                ));
                            }
                        }
                        _ => println!("  oracle no-op control not run"),
                    }
                } else if !unframed.is_empty() {
                    let _ = std::fs::remove_file(out);
                    die(format!(
                        "REFUSED and deleted {}: {} edit(s) change a string's length at a place \
                         with no enclosing skippable chunk to correct ({}). A body is parsed \
                         serially so that CAN be safe -- clearing a ghost's inline account id is \
                         -- but nothing in the file proves it, and the one time it was not safe \
                         the file still read back perfectly and validated to nothing. Re-run with \
                         a dedicated server (--server DIR or TM_SERVER) so the plain oracle can \
                         decide, or use a replacement of exactly the same byte length.",
                        out,
                        unframed.len(),
                        unframed.join(", ")
                    ));
                } else {
                    println!(
                        "  oracle no-op control NOT RUN (no dedicated server), but every edit kept \
                         its byte length or sits in a chunk whose size was corrected."
                    );
                }
            }
        }
        o => die(format!("unknown `ghost identity` operation {:?}", o)),
    }
}
