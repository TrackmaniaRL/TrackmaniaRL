# Historical tmrl-test backend snapshot

These three files preserve the exact bytes of the tmrl-test backend recorded by
its source manifest. They are research evidence, not a qualified deployable
runtime. The environment module retains historical Fall05 naming, but the
actual map guard and returned UID are for tmrl-test; it is not the Fall #05
backend. It requires separately obtained and explicitly configured native
assets, manifests, qualified engine build, a 22-element rotation observation,
and the original 10 ms/native event guard contract. No game files are included.

SHA-256 pins for the three files are checked by `test_contract.py`; they are not
a substitute for reviewing the backend or checking every transitive dependency.
The previous campaign qualification applies to its original frozen source and
assets, not to this checkout. A fresh bounded native qualification, exact
closure/clock/command/Goal validation, and separate Windows/TICK parity check
are required before production use. Import and contract tests do not start the
native game or establish controller performance or a world record.
