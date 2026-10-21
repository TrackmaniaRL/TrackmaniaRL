//! One argument parser for the whole crate.
//!
//! Before this there were three styles and twenty-four copies of the same
//! four-line closure:
//!
//! ```text
//! let flag = |n: &str| args.iter().position(|a| a == n)
//!     .and_then(|i| args.get(i + 1)).cloned();
//! ```
//!
//! copy-pasted across nine modules, plus a `parse_args(a, valueless: &[&str])`
//!   whose flag *schema* was a string literal at each call site, plus bespoke
//! `while i < argv.len()` loops. They disagreed in ways that mattered:
//!
//! * **Positional files were selected three different ways.** Filtering on a
//!   leading `--` made `--race 12759` look like a ghost called `12759` and the
//!   gate REFUSED it. Three modules learned that separately and fixed it
//!   separately. Here a flag's VALUE is consumed by the flag, so a positional
//!   is whatever is left, and there is one implementation to be right.
//! * **A typo in a value silently chose a default.** `--sort xyz` matched
//!   `"name" => Name, _ => Time`, so any misspelling quietly meant Time, and
//!   `--metric xyz` reached `.expect("bad --metric")` — a panic where a usage
//!   error belonged. `Args::enumerated` refuses an unrecognised value and
//!   names the alternatives.
//! * **Unknown flags were ignored.** `intg pair --kind X` parsed and dropped
//!   `--kind`; `intg dup` read `--server` and `--maps` and discarded both. A
//!   flag that does nothing is indistinguishable from a flag that failed.
//!   `Args::finish` refuses anything the command did not ask for.

use std::cell::RefCell;
use std::collections::BTreeMap;

pub struct Args {
    prog: String,
    vals: BTreeMap<String, Vec<String>>,
    switches: Vec<String>,
    pub positional: Vec<String>,
    seen: Vec<String>,
    asked: RefCell<Vec<String>>,
    bad: RefCell<Vec<String>>,
}

/// Parse `--flag value` / `--switch` / positionals.
///
/// `switches` names the flags that take no value; everything else spelled
/// `--x` consumes the next argument. `--x=y` is accepted for either.
pub fn parse(prog: &str, argv: &[String], switches: &[&str]) -> Args {
    let mut a = Args {
        prog: prog.to_string(),
        vals: BTreeMap::new(),
        switches: switches.iter().map(|s| s.to_string()).collect(),
        positional: Vec::new(),
        seen: Vec::new(),
        asked: RefCell::new(Vec::new()),
        bad: RefCell::new(Vec::new()),
    };
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        if let Some(rest) = arg.strip_prefix("--") {
            let (name, inline) = match rest.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (rest.to_string(), None),
            };
            a.seen.push(name.clone());
            if a.switches.iter().any(|s| *s == name) {
                a.vals.entry(name).or_default().push(inline.unwrap_or_else(|| "1".into()));
            } else if let Some(v) = inline {
                a.vals.entry(name).or_default().push(v);
            } else if i + 1 < argv.len() {
                i += 1;
                a.vals.entry(name).or_default().push(argv[i].clone());
            } else {
                a.bad.borrow_mut().push(format!("--{} needs a value", name));
            }
        } else {
            a.positional.push(arg.clone());
        }
        i += 1;
    }
    a
}

impl Args {
    pub fn has(&self, name: &str) -> bool {
        self.asked.borrow_mut().push(name.to_string());
        self.vals.contains_key(name)
    }

    pub fn one(&self, name: &str) -> Option<&str> {
        self.asked.borrow_mut().push(name.to_string());
        self.vals.get(name).and_then(|v| v.last()).map(|s| s.as_str())
    }

    /// Every occurrence of a repeatable flag, verbatim — no comma splitting.
    /// `many` flattens on commas, which is right for `--eps 1,2,3` and wrong
    /// for a flag whose own value contains commas (`--near X,Y,Z`, twice).
    pub fn repeated(&self, name: &str) -> Vec<String> {
        self.asked.borrow_mut().push(name.to_string());
        self.vals.get(name).cloned().unwrap_or_default()
    }

    pub fn many(&self, name: &str) -> Vec<String> {
        self.asked.borrow_mut().push(name.to_string());
        self.vals
            .get(name)
            .map(|v| v.iter().flat_map(|s| s.split(',')).map(|s| s.trim().to_string()).collect())
            .unwrap_or_default()
    }

    pub fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        match self.one(name) {
            None => default,
            Some(s) => match s.parse() {
                Ok(v) => v,
                Err(_) => {
                    self.bad.borrow_mut().push(format!("--{} wants a number, got {:?}", name, s));
                    default
                }
            },
        }
    }

    /// A value from a closed set. An unrecognised spelling is a usage error
    /// naming the alternatives — never a silent fall-through to a default.
    pub fn enumerated<T: Copy>(&self, name: &str, table: &[(&str, T)], default: T) -> T {
        match self.one(name) {
            None => default,
            Some(s) => match table.iter().find(|(k, _)| *k == s) {
                Some((_, v)) => *v,
                None => {
                    let names: Vec<&str> = table.iter().map(|(k, _)| *k).collect();
                    self.bad.borrow_mut().push(format!(
                        "--{} must be one of {}, got {:?}",
                        name,
                        names.join(" | "),
                        s
                    ));
                    default
                }
            },
        }
    }

    /// Every flag this command understands must have been asked for by now.
    /// Anything else, and any accumulated parse error, exits 2 with a usage
    /// message. Call it once, after reading the flags and before doing work.
    ///
    /// `--help` / `-h` are handled HERE, for every subcommand at once: usage
    /// on stdout, exit 0. No command asks for them, so before this they fell
    /// through to the unknown-flag path and every `tmtraj <cmd> --help`
    /// answered "unknown flag --help" while printing usage that lists it.
    pub fn finish(self, usage: &str) -> Args {
        if self.seen.iter().any(|s| s == "help" || s == "h") {
            print!("{}", usage);
            std::process::exit(0);
        }
        let known = self.asked.borrow().clone();
        for s in &self.seen {
            if !known.iter().any(|k| k == s) {
                self.bad.borrow_mut().push(format!(
                    "unknown flag --{}\n  (if this command DOES implement --{}, the bug is here: \
                     `finish` compares what was SEEN against what has been ASKED FOR SO FAR, so \
                     it must be called AFTER every `one`/`num`/`many`/`has` in the command, not \
                     before them)",
                    s, s
                ));
            }
        }
        let bad = self.bad.borrow().clone();
        if !bad.is_empty() {
            for b in &bad {
                eprintln!("{}: {}", self.prog, b);
            }
            eprint!("\n{}", usage);
            std::process::exit(2);
        }
        self
    }
}
