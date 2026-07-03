# BOG Object Formats v0

Canonical v0 files are deterministic JSON with sorted keys and compact separators. JSON is not described as binary; the schemas are designed so a future binary encoding can carry the same fields.

All objects include:

- `format`
- `schema_version`

Unknown required-object keys are rejected by strict validators.

## `.bogexe`

Format identifier: `bogexe`.

Contains `machine_version`, `entry_point`, `instructions`, `symbols`, and `program_hash`. The hash is over the canonical entry point and instruction stream. Tampering with instructions or hash is rejected.

## `.bogpkg`

Format identifier: `bogpkg`.

Contains `package_id` and process definitions. Each process embeds a validated `.bogexe`, initial position, priority, amplitude, phase, and requested resources.

## `.bogstate`

Format identifier: `bogstate`.

Contains kernel version, runtime state, and a state hash. Large floating arrays must not be pretty-printed inline; future field snapshots should use a JSON manifest plus compressed NumPy payload.

## `.boggraph`

Format identifier: `boggraph`.

Contains compiler configuration, typed nodes, typed saddles, edges, verification receipts, and graph hash.

## `.bogtrace`

Format identifier: `bogtrace`.

Contains an ordered receipt chain. Verification recomputes every receipt hash and checks every `previous_hash`. Tampering causes verification failure.
