# Public API and compatibility

[Documentation index](README.md) · [Python quickstart](python-quickstart.md)

Use the documented entry points when integrating TrackmaniaRL into another project.
An importable implementation module is not automatically a supported extension point.

| Surface | Intended use |
| --- | --- |
| `trackmaniarl.RunSpec` | Parse and validate run configuration |
| `trackmaniarl.resolve_run` | Construct configured components |
| `trackmaniarl.Trainer` | Local training and full-checkpoint resume |
| `trackmaniarl.__version__` | Inspect the installed package version |
| Exports in `trackmaniarl.core` | Runtime protocols, data, replay and training results |
| `trackmaniarl.project.scaffold.create_project` | Generate a separate application project |
| Component paths documented in the [SDK](sdk.md) | Select or replace models, learners and other components |

The [configuration reference](configuration.md) defines accepted configuration
fields. The [support matrix](../docs/support-status.md) describes supported
combinations. Package versions, RunSpec versions and checkpoint schema versions
are separate contracts; consult the [changelog](../CHANGELOG.md) before upgrading.
Exact resume additionally checks run and architecture fingerprints.

## Extension boundaries

Implement the protocols documented in the SDK and place custom code in your own
installed package. Select it using `package.module:ClassName` and component
`kwargs`. See the [examples](../examples/README.md) for a custom environment,
reward and encoder. Custom components do not require changes to the library.

`trackmaniarl.commands`, underscore-prefixed helpers and runtime implementation
modules are internal. Import public names through the facades above when available.
`trackmaniarl.experiments` contains research tooling;
their importability does not give them the stable SDK contract. Existing paths
remain available for historical configurations and reproductions.

## Resource ownership

A successfully resolved run belongs to the caller. Always close `run.logger` in
a `finally` block. Resolution cleans up loggers if construction fails. The trainer
owns environment resources it creates during training. The runnable examples
demonstrate this lifecycle.
