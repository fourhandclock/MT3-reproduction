# Experiment records

Historical stage packages and local `outputs/`, `data/` and `models/` paths are preserved because saved experiments bind code and artifacts to exact SHA256 values.

`stage19.snapshot.json` is a tracked copy of the completed Stage19 verification record. Its relative paths refer to the original workspace roots: artifact paths are under `outputs/fret19/`, and code paths are relative to the project root. The large artifacts remain local and ignored by Git; this snapshot records how to verify their identity.

Git tracks source, configuration, tests, documentation, small reference fixtures and compact experiment records. Git does not back up the ignored datasets, model weights or experiment outputs. Preserve those separately when moving this project.
