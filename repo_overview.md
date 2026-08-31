# repo_overview.md

Source: `get_overview()` on `src/paf` — the tool's own render.
File `imp=N` = dependency importance (in+out resolved edges, call=3, ref=1).
Dir line `imp: mean=M max=N` = mean & max importance over all descendant files.

---

## DEPTH 2 (folders + direct files)

```
Overview of /home/alice/projects/Personal/pydantic-agent-framework/src/paf
13 dirs, 67 files
file `imp=N` = dependency importance (in+out, call=3, ref=1); dir line: `imp: mean=N max=M` over its files
capabilities/  (62 files)  imp: mean=12.2 max=40
  browser/  (2 files)  imp: mean=12.0 max=13
  codebase_map/  (16 files)  imp: mean=14.1 max=30
  fs/  (2 files)  imp: mean=6.5 max=7
  git/  (2 files)  imp: mean=5.5 max=7
  memory/  (12 files)  imp: mean=14.2 max=28
  search/  (5 files)  imp: mean=11.4 max=17
  shell/  (3 files)  imp: mean=4.7 max=6
  skills/  (4 files)  imp: mean=10.0 max=13
  vault/  (4 files)  imp: mean=11.2 max=14
  visual/  (3 files)  imp: mean=6.3 max=8
  web/  (2 files)  imp: mean=7.5 max=8
  _markers.py  (38 lines)  imp=10
  _turns.py  (32 lines)  imp=13
  build.py  (667 lines)  imp=40
  compaction.py  (183 lines)  imp=12
  persistence.py  (104 lines)  imp=9
  profiles.py  (191 lines)  imp=18
  registry.py  (166 lines)  imp=18
observability/  (1 files)  imp: mean=5.0 max=5
  instrumentation.py  (140 lines)  imp=5
background.py  (157 lines)  imp=5
config.py  (117 lines)  imp=21
runtime.py  (550 lines)  imp=26
specs.py  (508 lines)  imp=22
```

---

## FULL

```
Overview of /home/alice/projects/Personal/pydantic-agent-framework/src/paf
13 dirs, 67 files
file `imp=N` = dependency importance (in+out, call=3, ref=1); dir line: `imp: mean=N max=M` over its files
capabilities/  (62 files)  imp: mean=12.2 max=40
  browser/  (2 files)  imp: mean=12.0 max=13
    _capability.py  (386 lines)  imp=11
    _toolset.py  (2815 lines)  imp=13
  codebase_map/  (16 files)  imp: mean=14.1 max=30
    _analyze.py  (311 lines)  imp=12
    _block.py  (59 lines)  imp=12
    _calls.py  (282 lines)  imp=8
    _capability.py  (90 lines)  imp=13
    _graph.py  (246 lines)  imp=12
    _overview.py  (88 lines)  imp=11
    _queries.py  (503 lines)  imp=10
    _rank.py  (179 lines)  imp=13
    _retrieval.py  (212 lines)  imp=18
    _rules.py  (18 lines)  imp=1
    _scan.py  (212 lines)  imp=24
    _schema.py  (127 lines)  imp=30
    _store.py  (339 lines)  imp=24
    _tokens.py  (180 lines)  imp=13
    _toolset.py  (486 lines)  imp=20
    cli.py  (53 lines)  imp=5
  fs/  (2 files)  imp: mean=6.5 max=7
    _capability.py  (86 lines)  imp=7
    _toolset.py  (513 lines)  imp=6
  git/  (2 files)  imp: mean=5.5 max=7
    _capability.py  (42 lines)  imp=7
    _toolset.py  (432 lines)  imp=4
  memory/  (12 files)  imp: mean=14.2 max=28
    _atomic.py  (51 lines)  imp=9
    _background.py  (317 lines)  imp=23
    _capability.py  (503 lines)  imp=28
    _consolidate.py  (631 lines)  imp=24
    _format.py  (64 lines)  imp=1
    _journal.py  (304 lines)  imp=14
    _render.py  (112 lines)  imp=10
    _store.py  (242 lines)  imp=7
    _toolset.py  (301 lines)  imp=9
    _view.py  (75 lines)  imp=16
    _watermark.py  (114 lines)  imp=9
    _window.py  (159 lines)  imp=21
  search/  (5 files)  imp: mean=11.4 max=17
    _capability.py  (108 lines)  imp=10
    _ranking.py  (128 lines)  imp=13
    _reader.py  (222 lines)  imp=10
    _render.py  (142 lines)  imp=7
    _toolset.py  (468 lines)  imp=17
  shell/  (3 files)  imp: mean=4.7 max=6
    _capability.py  (124 lines)  imp=5
    _output.py  (23 lines)  imp=3
    _toolset.py  (531 lines)  imp=6
  skills/  (4 files)  imp: mean=10.0 max=13
    _capability.py  (54 lines)  imp=7
    _corpus.py  (164 lines)  imp=12
    _point.py  (26 lines)  imp=13
    _toolset.py  (128 lines)  imp=8
  vault/  (4 files)  imp: mean=11.2 max=14
    _capability.py  (85 lines)  imp=11
    _markdown.py  (167 lines)  imp=7
    _store.py  (275 lines)  imp=14
    _toolset.py  (626 lines)  imp=13
  visual/  (3 files)  imp: mean=6.3 max=8
    _capability.py  (196 lines)  imp=8
    _store.py  (76 lines)  imp=5
    _toolset.py  (118 lines)  imp=6
  web/  (2 files)  imp: mean=7.5 max=8
    _capability.py  (40 lines)  imp=8
    _toolset.py  (378 lines)  imp=7
  _markers.py  (38 lines)  imp=10
  _turns.py  (32 lines)  imp=13
  build.py  (667 lines)  imp=40
  compaction.py  (183 lines)  imp=12
  persistence.py  (104 lines)  imp=9
  profiles.py  (191 lines)  imp=18
  registry.py  (166 lines)  imp=18
observability/  (1 files)  imp: mean=5.0 max=5
  instrumentation.py  (140 lines)  imp=5
background.py  (157 lines)  imp=5
config.py  (117 lines)  imp=21
runtime.py  (550 lines)  imp=26
specs.py  (508 lines)  imp=22
```
