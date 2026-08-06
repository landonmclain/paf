"""storage — reserved for the framework redo.

The old flat-file KV store (``Storage`` protocol + ``FlatFileStorage``) has been
removed: pydantic-ai (via the harness ``StepPersistence`` / ``SqliteStepStore``)
ships durable storage, and M1 embraces that rather than a custom protocol. A
persistence wrapper (create/restore/append) lands here during the framework
redo.
"""
