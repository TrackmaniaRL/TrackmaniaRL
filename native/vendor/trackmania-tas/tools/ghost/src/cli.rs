//! Argument handling for the `ghost` command line.
//!
//! Kept apart from the data path on purpose: `die` exits the process, which is
//! right for a CLI and wrong for a library, so nothing under `tape`,
//! `container`, `oracle` or `verify` may call it. Those return `Result`.

/// THE EXIT-CODE CONTRACT, for every tool in this workspace.
///
/// ```text
///   0  success
///   1  the operation ran and the answer is NO -- a gate refused, a
///      comparison disagreed, a verification failed. The tool worked.
///   2  usage error -- bad flags, missing arguments, a file that is not there
///   3  environment error -- no dedicated server, no engine, no network
/// ```
///
/// The distinction that matters is 1 vs 2. Before this was written down,
/// `die()` (exit 2) was used for both "you typed it wrong" and "this ghost is
/// not publishable", so a script could not tell a mistake from a verdict --
/// and the whole pipeline is scripted. `refuse()` is the verdict path.
///
/// Diagnostics go to stderr; data goes to stdout. A tool being piped must not
/// interleave the two.
pub fn die(m: impl AsRef<str>) -> ! {
    eprintln!("ghost: {}", m.as_ref());
    std::process::exit(2)
}

/// The operation ran and the answer is NO. Exit 1, not 2.
///
/// Use this for a gate that refuses, a verification that fails, a comparison
/// that disagrees -- anything where the tool did its job and the job's answer
/// is negative. A caller can then branch on the difference:
///
/// ```text
///   ghost verify FILE || case $? in
///     1) echo "not publishable" ;;
///     2) echo "I called it wrong" ;;
///   esac
/// ```
pub fn refuse(m: impl AsRef<str>) -> ! {
    eprintln!("ghost: {}", m.as_ref());
    std::process::exit(1)
}

pub fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}

pub fn need<'a>(a: &'a [String], name: &str) -> &'a str {
    flag(a, name).unwrap_or_else(|| die(format!("missing {} VALUE", name)))
}

pub fn has(a: &[String], name: &str) -> bool {
    a.iter().any(|x| x == name)
}

pub fn num(a: &[String], name: &str) -> Option<i64> {
    flag(a, name).map(|v| v.parse().unwrap_or_else(|_| die(format!("{} wants a number", name))))
}

