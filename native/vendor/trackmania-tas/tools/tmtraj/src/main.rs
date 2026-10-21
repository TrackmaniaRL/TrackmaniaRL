fn main() {
    // --version / -V, before the library takes over. Compile-time only:
    // CARGO_PKG_* come from Cargo.toml (which inherits the one workspace
    // version) and TAS_BUILD is the git hash the release build sets.
    if std::env::args().any(|x| x == "--version" || x == "-V") {
        println!(
            "{} {} ({})",
            option_env!("CARGO_BIN_NAME").unwrap_or(env!("CARGO_PKG_NAME")),
            env!("CARGO_PKG_VERSION"),
            option_env!("TAS_BUILD").unwrap_or("dev")
        );
        std::process::exit(0);
    }
    tmtraj::run();
}
