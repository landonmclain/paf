"""Approval flow P3: shell commands are understood, and destructive ones ask.

The old check looked at the first word only: `ls; rm -rf x`, `sudo rm`, `bash -c`,
`xargs rm` and `find -delete` all passed while a plain `rm` was refused, which
taught the model to work around it. Now every simple command is analysed, a
policy decides allow / approve / block, the person sees a preview of what would
be deleted, and the workarounds ask too.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ApprovalRequired, SkipToolExecution
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from paf.approvals import (
    ApprovalBroker,
    ApprovalChannel,
    ApprovalRequested,
    bind_channel,
)
from paf.capabilities.build import _build_shell
from paf.capabilities.shell import Shell
from paf.capabilities.shell._analyze import analyze
from paf.capabilities.shell._capability import COMMAND_POLICY_INSTRUCTIONS
from paf.capabilities.shell._command_policy import evaluate, parse_command_policy
from paf.capabilities.shell._preview import preview
from paf.specs import _spec_shell_config


def _run(coro):
    return asyncio.run(coro)


LENIENT = {
    "default": "allow",
    "unparseable": "approve",
    "unknown_scripts": "allow",
    "rules": [
        {
            "effects_any": ["delete"],
            "targets_any": ["/", "/*", "~", "~/", "$HOME", "/home", "/etc", "/usr"],
            "action": "block",
            "why": "deletes the root, your home, or a system directory",
        },
        {
            "effects_any": ["delete", "discard", "force_push", "disk"],
            "action": "approve",
        },
        {
            "commands": ["chmod", "chown"],
            "flags_any": ["-R", "--recursive"],
            "action": "approve",
        },
    ],
}


# --- the analyzer ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("command", "name", "effects", "via"),
    [
        ("rm -rf build", "rm", {"delete"}, ()),
        ("ls; rm -rf ~/x", "rm", {"delete"}, ()),
        ("echo ok && rm -rf ~/x", "rm", {"delete"}, ()),
        ("sudo rm -rf x", "rm", {"delete", "privileged"}, ("sudo",)),
        ("find . -name '*.pyc' -delete", "find", {"delete"}, ()),
        ("bash -c 'rm -rf x'", "rm", {"delete"}, ("bash -c",)),
        ("git ls-files | xargs rm", "rm", {"delete"}, ("xargs",)),
        ("X=$(rm z)", "rm", {"delete"}, ()),
        ("find . -exec rm {} \\;", "rm", {"delete"}, ("find -exec",)),
        (
            "python3 -c \"import shutil; shutil.rmtree('build')\"",
            "python3",
            {"delete"},
            (),
        ),
        (
            "env FOO=1 timeout 5 nice -n 10 rm -f a",
            "rm",
            {"delete"},
            ("env", "timeout", "nice"),
        ),
        ("git clean -fdx", "git", {"delete"}, ()),
        ("git -C repo clean -fd", "git", {"delete"}, ()),
        ("git rm a.txt", "git", {"delete"}, ()),
        ("git reset --hard HEAD~1", "git", {"discard"}, ()),
        ("git push --force origin main", "git", {"force_push"}, ()),
        ("dd if=/dev/zero of=/dev/sda", "dd", {"disk"}, ()),
        ("chmod -R 777 .", "chmod", {"permissions"}, ()),
        ("./clean.sh", "clean.sh", {"script"}, ()),
        ("bash deploy.sh", "bash", {"script"}, ()),
    ],
)
def test_analyzer_finds_the_command_that_matters(command, name, effects, via):
    cmds = [c for c in analyze(command).commands if c.effects]
    assert cmds, command
    hit = cmds[0]
    assert hit.name == name and hit.effects == frozenset(effects) and hit.via == via


@pytest.mark.parametrize(
    "command",
    [
        'echo "rm -rf /"',
        "grep rm notes.txt",
        "git rm --cached a.txt",
        "git clean -n",
        "git push origin main",
        "git status",
        "ls -la",
        "cat file | tee out.txt",
        "python -m pytest -q",
    ],
)
def test_analyzer_leaves_look_alikes_alone(command):
    a = analyze(command)
    assert a.effects == frozenset() and a.problems == ()


@pytest.mark.parametrize(
    ("command", "problem"),
    [
        ('eval "$CMD"', "eval"),
        ("$CMD arg", "computed at run time"),
        ("rm -rf (", "could not parse"),
    ],
)
def test_analyzer_reports_what_it_cannot_see(command, problem):
    assert any(problem in p for p in analyze(command).problems)


def test_targets_and_flags():
    (cmd,) = analyze("rm -rf -- build 'my dir' -x").commands
    assert cmd.targets == ("build", "my dir", "-x")
    assert {"-r", "-f", "-rf"} <= cmd.flags


# --- the policy ----------------------------------------------------------------------------
def _decide(command: str, cwd: Path = Path("/tmp/project"), policy=LENIENT) -> str:
    return evaluate(parse_command_policy(policy), analyze(command), cwd).action


@pytest.mark.parametrize(
    ("command", "action"),
    [
        ("ls -la", "allow"),
        ("pytest -q", "allow"),
        ("rm -rf build", "approve"),
        ("ls; rm -rf build", "approve"),  # compound: strictest wins
        ("sudo rm -rf build", "approve"),
        ("find . -name '*.pyc' -delete", "approve"),
        ("python3 -c \"import os; os.remove('a')\"", "approve"),
        ("git ls-files | xargs rm", "approve"),
        ("git clean -fdx", "approve"),
        ("git reset --hard", "approve"),
        ("git push -f origin main", "approve"),
        ("chmod -R 755 .", "approve"),
        ("chmod 644 a.txt", "allow"),
        ("rm -rf /", "block"),
        ("rm -rf /*", "block"),
        ("rm -rf ~", "block"),
        ("rm -rf ~/", "block"),
        ("rm -rf $HOME", "block"),
        ("sudo rm -rf /etc", "block"),
        ("echo hi && rm -rf ~", "block"),  # block beats approve beats allow
        ('eval "$X"', "approve"),  # unparseable -> approve
        ("./clean.sh", "allow"),  # unknown scripts allowed on the lenient profile
        ('echo "rm -rf /"', "allow"),
    ],
)
def test_lenient_policy_decisions(command, action):
    assert _decide(command) == action


def test_unknown_scripts_can_be_made_to_ask():
    strict = {**LENIENT, "unknown_scripts": "approve"}
    assert _decide("./clean.sh", policy=strict) == "approve"


def test_the_reason_names_the_command_and_why():
    d = evaluate(
        parse_command_policy(LENIENT),
        analyze("ls && sudo rm -rf build"),
        Path("/tmp/p"),
    )
    assert d.action == "approve"
    assert (
        "`rm -rf build`" in d.reason
        and "via sudo" in d.reason
        and "deletes files" in d.reason
    )
    d = evaluate(parse_command_policy(LENIENT), analyze("rm -rf ~"), Path("/tmp/p"))
    assert "deletes the root, your home" in d.reason


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"default": "maybe"}, "must be one of"),
        (
            {"rules": [{"action": "approve", "effects_any": ["explode"]}]},
            "unknown effects",
        ),
        ({"rules": [{"action": "approve", "commands": "rm"}]}, "non-empty list"),
        ({"rules": [{"action": "approve", "bogus": 1}]}, "unknown keys"),
        ({"bogus": 1}, "unknown keys"),
        ({"rules": ["rm"]}, "must be a mapping"),
    ],
)
def test_malformed_policies_refuse_to_load(config, match):
    with pytest.raises(ValueError, match=match):
        parse_command_policy(config)


def test_absent_policy_is_none():
    assert parse_command_policy(None) is None and parse_command_policy({}) is None


# --- the preview -----------------------------------------------------------------------------
def _tree(root: Path) -> None:
    (root / "build" / "sub").mkdir(parents=True)
    for i in range(5):
        (root / "build" / "sub" / f"f{i}.o").write_bytes(b"x" * 1000)
    (root / "dist").mkdir()
    (root / "dist" / "pkg.whl").write_bytes(b"y" * 3000)
    (root / "src.py").write_text("print(1)\n")


def test_preview_counts_what_would_be_deleted(tmp_path):
    _tree(tmp_path)
    d = evaluate(parse_command_policy(LENIENT), analyze("rm -rf build dist"), tmp_path)
    p = preview(d.deciding, tmp_path)
    assert "build" in p["deletes"] and "dist" in p["deletes"]
    assert "6 files" in p["deletes"] and "7.8 KB" in p["deletes"]


def test_preview_flags_missing_paths_empty_globs_and_outside_paths(tmp_path):
    _tree(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "x").write_text("x")
    cmd = f"rm -rf nope/ *.tmp {outside}"
    d = evaluate(parse_command_policy(LENIENT), analyze(cmd), tmp_path)
    p = preview(d.deciding, tmp_path)
    assert p["not found"] == "nope/" and p["matched nothing"] == "*.tmp"
    assert "outside" in p["warning"]


def test_preview_counts_files_tracked_by_git(tmp_path):
    _tree(tmp_path)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "add", "src.py"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "x"], cwd=tmp_path, check=True)
    d = evaluate(
        parse_command_policy(LENIENT), analyze("rm src.py build/sub/f0.o"), tmp_path
    )
    p = preview(d.deciding, tmp_path)
    assert "1 file(s) tracked by git" in p["warning"]


def test_preview_explains_what_it_cannot_count(tmp_path):
    for command, text in (
        ("find . -name '*.pyc' -delete", "find deletes whatever"),
        ("git clean -fdx", "with -n to list them"),
        ("git ls-files | xargs rm", "computed at run time"),
    ):
        d = evaluate(parse_command_policy(LENIENT), analyze(command), tmp_path)
        assert text in preview(d.deciding, tmp_path)["note"], command


# --- the shell hook --------------------------------------------------------------------------------
def _ctx(approved=False) -> RunContext:
    return RunContext(
        deps=None, model=TestModel(), usage=RunUsage(), tool_call_approved=approved
    )


def _hook(shell: Shell, command: str, approved=False, tool="run_command"):
    call = ToolCallPart(tool, {"command": command}, tool_call_id="call-1")
    return _run(
        shell.before_tool_execute(
            _ctx(approved),
            call=call,
            tool_def=ToolDefinition(name=tool),
            args={"command": command},
        )
    )


def _shell(tmp_path, policy=LENIENT) -> Shell:
    return Shell(cwd=tmp_path, command_policy=parse_command_policy(policy))


def test_no_policy_keeps_the_legacy_shell(tmp_path):
    shell = Shell(cwd=tmp_path)
    assert _hook(shell, "rm -rf build") == {
        "command": "rm -rf build"
    }  # the deny list decides later
    assert "rm" in shell.denied_commands
    assert shell.get_instructions() is None


def test_a_policy_replaces_the_first_word_deny_list(tmp_path):
    shell = _shell(tmp_path)
    assert shell.denied_commands == []
    assert shell.get_instructions() == [COMMAND_POLICY_INSTRUCTIONS]


def test_allowed_commands_pass(tmp_path):
    assert _hook(_shell(tmp_path), "ls -la") == {"command": "ls -la"}


def test_blocked_commands_are_refused_before_they_run(tmp_path):
    with pytest.raises(SkipToolExecution) as exc:
        _hook(_shell(tmp_path), "rm -rf ~")
    assert "refused by the agent's command policy" in exc.value.result


def test_approve_with_nobody_attached_is_refused(tmp_path):
    with pytest.raises(SkipToolExecution) as exc:
        _hook(_shell(tmp_path), "rm -rf build")
    assert "no one is attached" in exc.value.result


def test_approve_asks_with_reason_and_preview(tmp_path):
    _tree(tmp_path)
    channel = ApprovalChannel(emit=lambda ev: None)

    async def scenario():
        with bind_channel(channel):
            call = ToolCallPart(
                "run_command", {"command": "rm -rf build"}, tool_call_id="call-1"
            )
            with pytest.raises(ApprovalRequired):
                await _shell(tmp_path).before_tool_execute(
                    _ctx(),
                    call=call,
                    tool_def=ToolDefinition(name="run_command"),
                    args={"command": "rm -rf build"},
                )

    _run(scenario())
    req = channel.new_request(
        tool_call_id="call-1", tool_name="run_command", args={"command": "rm -rf build"}
    )
    assert "deletes files" in req.reason and "5 files" in req.preview["deletes"]


def test_an_approved_call_proceeds(tmp_path):
    assert _hook(_shell(tmp_path), "rm -rf build", approved=True) == {
        "command": "rm -rf build"
    }


def test_background_commands_are_checked_too(tmp_path):
    with pytest.raises(SkipToolExecution):
        _hook(_shell(tmp_path), "rm -rf build", tool="start_command")


# --- end to end: a real agent, the real shell, a real rm --------------------------------------------
def _agent_run(tmp_path, commands, decide):
    """Run the model's `commands` through a real Shell; `decide(request)` answers."""
    queue = [ToolCallPart("run_command", {"command": c}) for c in commands]

    def respond(messages, info):
        return (
            ModelResponse(parts=[queue.pop(0)])
            if queue
            else ModelResponse(parts=[TextPart("done")])
        )

    shell = _shell(tmp_path)
    agent = Agent(FunctionModel(respond), capabilities=[shell, ApprovalBroker()])
    asked: list = []

    def emit(ev):
        if isinstance(ev, ApprovalRequested):
            asked.append(ev.request)
            answer = decide(ev.request)
            if answer is not None:
                channel.resolve(ev.request.id, *answer)

    channel = ApprovalChannel(emit=emit, timeout=5) if decide is not None else None

    async def go():
        with bind_channel(channel):
            return await agent.run("clean up")

    result = _run(go())
    returns = [
        str(p.content)
        for m in result.all_messages()
        for p in m.parts
        if p.part_kind == "tool-return"
    ]
    return asked, returns


def test_e2e_approved_rm_actually_runs(tmp_path):
    _tree(tmp_path)
    asked, _ = _agent_run(tmp_path, ["rm -rf build"], decide=lambda r: (True,))
    assert len(asked) == 1 and "5 files" in asked[0].preview["deletes"]
    assert not (tmp_path / "build").exists() and (tmp_path / "dist").exists()


def test_e2e_denied_rm_does_not_run_and_the_model_hears_why(tmp_path):
    _tree(tmp_path)
    asked, returns = _agent_run(tmp_path, ["rm -rf build"], decide=lambda r: (False,))
    assert (tmp_path / "build").exists()
    assert "denied" in returns[0]


def test_e2e_with_nobody_attached_nothing_is_deleted(tmp_path):
    _tree(tmp_path)
    asked, returns = _agent_run(tmp_path, ["rm -rf build"], decide=None)
    assert asked == [] and (tmp_path / "build").exists()
    assert "no one is attached" in returns[0]


def test_e2e_the_workaround_asks_too(tmp_path):
    _tree(tmp_path)
    asked, _ = _agent_run(
        tmp_path, ["find build -name '*.o' -delete"], decide=lambda r: (False,)
    )
    assert len(asked) == 1 and "find" in asked[0].reason
    assert len(list((tmp_path / "build" / "sub").iterdir())) == 5


def test_e2e_ordinary_commands_never_ask(tmp_path):
    _tree(tmp_path)
    asked, returns = _agent_run(tmp_path, ["ls build"], decide=lambda r: (False,))
    assert asked == [] and "sub" in returns[0]


# --- builder and spec ---------------------------------------------------------------------------------
def test_builder_reads_the_policy_and_rejects_a_bad_one(tmp_path):
    shell = _build_shell(None, {"cwd": str(tmp_path), "command_policy": LENIENT})
    assert shell.command_policy is not None and shell.denied_commands == []
    with pytest.raises(ValueError):
        _build_shell(None, {"cwd": str(tmp_path), "command_policy": {"default": "nah"}})
    legacy = _build_shell(None, {"cwd": str(tmp_path)})
    assert legacy.command_policy is None and "rm" in legacy.denied_commands


def test_spec_reader_passes_the_block_through(tmp_path):
    spec = tmp_path / "a.yaml"
    spec.write_text(
        "shell:\n  cwd: ~/x\n  command_policy:\n    default: allow\n"
        "    rules:\n      - effects_any: [delete]\n        action: approve\n"
    )
    cfg = _spec_shell_config(spec)
    assert cfg["command_policy"]["rules"][0]["action"] == "approve"
