# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor ``[[steps]]`` tasks run as turns: one agent session per turn on one sandbox, then the roll-up."""

import io
import json
import logging
import posixpath
import tarfile
from pathlib import Path

import pytest

from environment_servers.harbor.app import (
    NO_TURN_RESULT_KIND,
    HarborEpisodeRequest,
    HarborTurnResult,
    min_reward_failure,
    roll_up_turns,
)
from environment_servers.harbor.tests.conftest import TASK_TOML, FakeSandbox, agent_response, make_server
from nemo_gym.tasks.harbor import load_task
from nemo_gym.tasks.harbor.materialize import STEP_METADATA_KEY, materialize_task


# The top-level instruction.md `write_task` writes: in a multi-step task it may hold what the agent must not see.
HIDDEN_SPEC = "Create hello.txt"


def write_steps(task_dir: Path, steps: dict[str, str]) -> None:
    """``steps`` maps each step name to its instruction; every step gets a test.sh of its own."""
    for name, instruction in steps.items():
        (task_dir / "steps" / name / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "steps" / name / "instruction.md").write_text(instruction)
        (task_dir / "steps" / name / "tests" / "test.sh").write_text(
            "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
        )


def multi_turn_server(tmp_path, monkeypatch, steps_toml: str, steps: dict[str, str], *, head: str = "", **config):
    """``head`` holds top-level keys, which TOML only reads before the first table."""
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, steps)
    return make_server(tmp_path, monkeypatch, toml=head + TASK_TOML + steps_toml, **config)


def rewards_per_turn(server, sandbox: FakeSandbox, rewards: list[str], *, name: str = "reward.txt") -> None:
    """Make the verifier of the n-th turn write ``rewards[n]``: each activation sets what its verify reads."""
    pending = list(rewards)

    def next_reward():
        sandbox.verifier_files = {name: pending.pop(0)}

    server.server_client.turn_hooks.append(next_reward)


def request_for(task, rollout: str = "r1") -> HarborEpisodeRequest:
    return HarborEpisodeRequest.model_validate(
        {"episode_id": {"rollout_id": rollout}, "task": materialize_task(task, "ds").model_dump(mode="json")}
    )


def uploaded_files(sandbox: FakeSandbox, name: str) -> list[bytes]:
    """The contents of ``name`` in every archive the server uploaded, in upload order."""
    found: list[bytes] = []
    for data in sandbox.uploaded_archives:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and posixpath.basename(member.name) == name:
                    found.append(tar.extractfile(member).read())
    return found


THREE_STEPS = """
[[steps]]
name = "plan"
[steps.agent]
timeout_sec = 60.0

[[steps]]
name = "build"
[steps.verifier]
timeout_sec = 30.0
env = { STEP = "build" }

[[steps]]
name = "check"
"""


async def test_each_step_is_a_turn_in_a_fresh_session_on_one_sandbox(tmp_path, monkeypatch):
    server, task, sandbox, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        THREE_STEPS,
        {"plan": "Agree on a plan.", "build": "Build it.", "check": "Hand off."},
        head='multi_step_reward_strategy = "final"\n',
    )
    rewards_per_turn(server, sandbox, ["1", "1", "0"])

    response = await server.run_request(request_for(task))

    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build", "check"]
    assert [turn.reward for turn in result.turns] == [1.0, 1.0, 0.0]
    # final: the last turn decides, even though earlier turns passed.
    assert (result.reward, result.turn_reward_strategy, result.stopped_after_turn) == (0.0, "final", None)
    # One sandbox for the whole episode, one agent session per turn.
    assert len(created) == 1
    seeds = server.server_client.seed_bodies()
    assert len({seed["agent_session_id"] for seed in seeds}) == 3
    assert len({str(seed["sandbox_access"]) for seed in seeds}) == 1
    # Each turn sends its own step's instruction and budget; the top-level instruction never leaves the server.
    turns = [
        call[2]["json"].model_dump(mode="json", exclude_unset=True)
        for call in server.server_client.calls
        if call[1].endswith("/v1/responses")
    ]
    assert [turn["input"][0]["content"] for turn in turns] == ["Agree on a plan.", "Build it.", "Hand off."]
    assert [turn["metadata"]["agent_timeout_sec"] for turn in turns] == ["60.0", "300.0", "300.0"]
    assert [turn["metadata"][STEP_METADATA_KEY] for turn in turns] == ["plan", "build", "check"]
    assert HIDDEN_SPEC not in str(server.server_client.calls)
    # The build step's verifier runs with its own timeout and environment over the task's.
    # The chmod of test.sh before each run is its own exec; the verifier's run is the one under `timeout`.
    build_verify = [e for e in sandbox.execs if "test.sh" in e["command"] and "timeout" in e["command"]][1]
    assert "timeout -s KILL 30 " in build_verify["command"] and build_verify["env"] == {"STEP": "build"}


async def test_mean_averages_the_turns_that_produced_a_result(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, ["1", "0.5"])

    result = (await server.run_request(request_for(task))).result

    assert result.turn_reward_strategy == "mean"
    assert result.reward == 0.75 and result.verifier_rewards == {"reward": 0.75}
    assert result.reward_components is None and all(turn.reward_components is None for turn in result.turns)


async def test_mean_averages_the_components_every_turn_named(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, [json.dumps({"x": 1.0, "y": 0.0}), json.dumps({"x": 0.0})], name="reward.json")

    result = (await server.run_request(request_for(task))).result

    assert [turn.reward_components for turn in result.turns] == [{"x": 1.0, "y": 0.0}, {"x": 0.0}]
    assert result.reward == 0.5 and result.reward_components == {"x": 0.5, "y": 0.0}
    assert sum(result.reward_components.values()) == result.reward


async def test_the_harness_trajectory_rides_on_each_turn_and_the_last_one_on_the_result(tmp_path, monkeypatch):
    """Every turn's session close carries the harness's own account of that turn; the episode forwards the last
    one in ``ng_trajectory``, the slot rollout collection reads, as it forwards the last turn's observations."""
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, ["1", "1", "1", "1"])
    server.server_client.trajectory = {
        "task_id": "hello",
        "rollout_id": "r1",
        "turns": [
            {
                "invocation_id": "harness_1",
                "task_id": "hello",
                "rollout_id": "r1",
                "turn_no": 1,
                "timestamp": 1.0,
                "question": "do it",
                "answer": [],
                "step_count": 2,
                "model_calls": [],
            }
        ],
    }

    result = (await server.run_request(request_for(task))).result

    assert [turn.trajectory.turns[0].step_count for turn in result.turns] == [2, 2]
    assert result.ng_trajectory is result.turns[-1].trajectory
    dumped = result.model_dump(mode="json")
    assert dumped["ng_trajectory"]["turns"][0]["step_count"] == 2
    # A harness that returns no trajectory leaves the key out of each turn, as it does on the result.
    server.server_client.trajectory = None
    dumped = (await server.run_request(request_for(task))).result.model_dump(mode="json")
    assert "trajectory" not in dumped["turns"][0] and "ng_trajectory" not in dumped


async def test_a_missed_min_reward_stops_the_remaining_turns(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\nmin_reward = 1.0\n[[steps]]\nname = "build"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    rewards_per_turn(server, sandbox, ["0"])

    result = (await server.run_request(request_for(task))).result

    assert [turn.name for turn in result.turns] == ["plan"]
    assert result.stopped_after_turn == "plan" and result.reward == 0.0
    assert sum(path.endswith("/v1/responses") for path in server.server_client.paths()) == 1


async def test_a_turn_without_a_reward_file_ends_the_episode_measured(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    sandbox.verifier_files = {}

    result = (await server.run_request(request_for(task))).result

    assert [turn.verifier_rewards for turn in result.turns] == [None]
    assert (result.reward, result.mask_sample, result.turns_without_verifier_result) == (0.0, False, 1)
    assert result.failure_kind == "harbor:missing_reward"


async def test_rows_whose_turns_differ_from_the_task_fail_terminally(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    request = request_for(task)
    request.task.task_input.turns.pop()

    response = await server.run_request(request)

    assert response.failure.terminal and "do not match the steps" in response.failure.failure_reason


async def test_each_turn_seeds_its_own_budget_and_hands_its_own_transcript_to_the_verifier(tmp_path, monkeypatch):
    """The single-turn protocol holds per turn: the harness is seeded with the step's agent budget, and the
    verifier reads an ATIF transcript of that turn's own request, so the hidden top-level instruction never
    reaches /logs/agent/trajectory.json either."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, THREE_STEPS, {"plan": "Agree on a plan.", "build": "Build it.", "check": "Hand off."}
    )

    response = await server.run_request(request_for(task))

    result = response.result
    assert result is not None, response.failure
    assert [seed["agent_timeout_sec"] for seed in server.server_client.seed_bodies()] == [60.0, 300.0, 300.0]
    transcripts = [json.loads(data) for data in uploaded_files(sandbox, "trajectory.json")]
    assert [steps["steps"][0]["source"] for steps in transcripts] == ["user"] * 3
    assert [steps["steps"][0]["message"] for steps in transcripts] == ["Agree on a plan.", "Build it.", "Hand off."]
    assert HIDDEN_SPEC not in json.dumps(transcripts)
    # Each turn's verifier run reports as a single turn's does, under its own logs folder.
    assert [turn.verifier_network for turn in result.turns] == ["public"] * 3
    assert len({turn.verifier_logs_dir for turn in result.turns}) == 3
    assert all((Path(turn.verifier_logs_dir) / "trajectory.json").is_file() for turn in result.turns)


async def test_the_task_deadline_counts_every_turn(tmp_path, monkeypatch):
    """Each step is a harness session, the hooks and a verifier run of its own; the episode budget sums them."""
    server, task, _, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        THREE_STEPS,
        {"plan": "P", "build": "B", "check": "C"},
        harness_setup_allowance_s=10,
        verifier_grace_s=5,
    )

    # plan: agent 60 (its own), verifier 120 (the task's); build: 300 and 30; check: 300 and 120.
    per_turn = (10 + 60 + 120 + 5) + (10 + 300 + 30 + 5) + (10 + 300 + 120 + 5)
    assert server._post_seed_budget_s(task) == pytest.approx(per_turn)
    assert server._task_budget_s(task) == pytest.approx(server.config.sandbox_ready_timeout_s + per_turn)


async def test_the_task_deadline_ends_an_overrunning_turn(tmp_path, monkeypatch):
    steps = """
[[steps]]
name = "plan"
[steps.agent]
timeout_sec = 0.05
[steps.verifier]
timeout_sec = 0.05

[[steps]]
name = "build"
[steps.agent]
timeout_sec = 0.05
[steps.verifier]
timeout_sec = 0.05
"""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"}, harness_setup_allowance_s=0, verifier_grace_s=0
    )
    assert server._post_seed_budget_s(task) == pytest.approx(0.2)
    server.server_client.slow_path = "/ng-rollout/r1/v1/responses"

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure.stage == "agent" and failure.failure_kind == "harbor:task_deadline" and not failure.terminal
    assert "time budget of task 'hello'" in failure.failure_reason and "over 2 turn(s)" in failure.failure_reason
    assert not any("test.sh" in call["command"] for call in sandbox.execs)
    # The final unwind still closed the agent session and stopped the sandbox.
    assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
    assert sandbox.stopped and not server._sandboxes and server._active == set()

    # The same slow harness finishes both turns when the run ignores task timeouts.
    server.config.enforce_task_timeouts = False
    server.server_client.slow_path = "/ng-rollout/r2/v1/responses"
    result = (await server.run_request(request_for(task, rollout="r2"))).result
    assert [turn.name for turn in result.turns] == ["plan", "build"] and result.reward == 1.0


@pytest.mark.parametrize("provider", ["docker", "opensandbox"])
async def test_a_step_verifier_user_with_a_group_follows_the_provider_at_seed(tmp_path, monkeypatch, provider):
    """Review (round 24): a step's `[steps.verifier].user = "name:group"` reached `_run_shared_verifier` as written
    through `_turn_verifier_settings` on every provider, while the task-level `[verifier].user` pair had been refused
    at seed (`_refuse_verifier_group`) where only Docker applies a group. Every step's pair is refused the same way
    now: at seed, before the sandbox exists and the turns before that step are paid for. The refusal names the step
    that carries the pair (repair round 4), as the loader's per-step refusals do, so the author of a six-step task need
    not search every `[steps.verifier]` table for it."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[steps.verifier]\nuser = "tester:grp"\n'
    server, task, sandbox, created = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    assert task.config.verifier.user is None and task.turns[1].config.verifier.user == "tester:grp"
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {provider: {}}}
    )

    response = await server.run_request(request_for(task))

    runs = [call for call in sandbox.execs if "bash /tests/test.sh" in call["command"]]
    if provider == "docker":
        assert response.failure is None, response.failure
        assert [turn.name for turn in response.result.turns] == ["plan", "build"]
        # The first turn grades as the task's verifier user; the second as the step's own pair, as written.
        assert [run["user"] for run in runs] == [None, "tester:grp"]
    else:
        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        # The step that carries the pair is named, right after the task id; the task's own user is not blamed.
        assert "hello step 'build': user 'tester:grp' names group 'grp'" in failure.failure_reason
        assert "hello: user" not in failure.failure_reason and "'plan'" not in failure.failure_reason
        assert "'opensandbox'" in failure.failure_reason and "switches user by name only" in failure.failure_reason
        # Nothing ran: no sandbox, no agent session, no turn.
        assert created == [] and sandbox.execs == [] and server.server_client.seed_bodies() == []
        assert runs == [] and response.result is None


def solution_uploads(sandbox: FakeSandbox) -> list[dict]:
    return [c for c in sandbox.execs if "tar -xzf" in c["command"] and "-C /solution" in c["command"]]


def write_step_solution(task_dir: Path, name: str, script: str) -> None:
    (task_dir / "steps" / name / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "steps" / name / "solution" / "solve.sh").write_text(script)


async def test_the_oracle_gets_each_steps_own_solution_before_its_turn(tmp_path, monkeypatch):
    """G7 round 24, attempt 4: the MemoryArena task keeps its reference solutions in `steps/<name>/solution/` and
    the server uploaded the task's top-level `solution/` once at seed, so the oracle had nothing to run. Harbor
    0.22.0's oracle resolves the step's own folder per step and falls back to the task's (`agents/oracle.py`
    `_resolve_solution_paths`); the server now places that folder at /solution before each turn, emptying what the
    previous turn left, so each turn's `/solution/solve.sh` is its step's and a step without any leaves none."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[[steps]]\nname = "check"\n'
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"plan": "P", "build": "B", "check": "C"})
    write_step_solution(task_dir, "plan", "#!/bin/bash\necho plan\n")
    write_step_solution(task_dir, "check", "#!/bin/bash\necho check\n")
    # No top-level solution/: `build` has none anywhere; the loader reads the folder as it is.
    server, task, sandbox, _ = make_server(
        tmp_path, monkeypatch, toml=TASK_TOML + steps, agent_implementation="harbor_oracle_agent"
    )
    assert not task.has_solution and not (task.path / "solution").exists()
    uploads_per_turn: list[int] = []
    server.server_client.turn_hooks.append(lambda: uploads_per_turn.append(len(solution_uploads(sandbox))))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build", "check"]
    # Nothing at seed; one upload before the first turn and before the third, none for the step without a solution.
    assert uploads_per_turn == [1, 1, 2]
    assert [data.decode() for data in uploaded_files(sandbox, "solve.sh")] == [
        "#!/bin/bash\necho plan\n",
        "#!/bin/bash\necho check\n",
    ]
    uploads = solution_uploads(sandbox)
    assert all(upload["user"] == "root" and "chmod -R a+rX /solution" in upload["command"] for upload in uploads)
    # /solution is emptied before every turn, as root, so `build` runs with no previous step's script in place.
    resets = [i for i, c in enumerate(sandbox.execs) if c["command"] == "rm -rf -- /solution"]
    assert len(resets) == 3 and all(sandbox.execs[i]["user"] == "root" for i in resets)
    upload_indexes = [i for i, c in enumerate(sandbox.execs) if c in uploads]
    assert resets[0] < upload_indexes[0] < resets[1] < resets[2] < upload_indexes[1]

    # The task's top-level solution/ stands in for a step without its own, as Harbor's oracle falls back to it.
    (task.path / "solution").mkdir()
    (task.path / "solution" / "solve.sh").write_text("#!/bin/bash\necho task\n")
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    assert task.has_solution
    sandbox.execs.clear()
    sandbox.uploaded_archives.clear()
    response = await server.run_request(request_for(task, rollout="r2"))
    assert response.failure is None, response.failure
    assert [data.decode() for data in uploaded_files(sandbox, "solve.sh")] == [
        "#!/bin/bash\necho plan\n",
        "#!/bin/bash\necho task\n",
        "#!/bin/bash\necho check\n",
    ]


async def test_other_harnesses_never_get_a_steps_solution(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"plan": "P", "build": "B"})
    write_step_solution(task_dir, "plan", "#!/bin/bash\necho plan\n")
    server, task, sandbox, _ = make_server(
        tmp_path, monkeypatch, toml=TASK_TOML + steps, agent_implementation="hermes_agent"
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert not any("/solution" in c["command"] for c in sandbox.execs)
    assert uploaded_files(sandbox, "solve.sh") == []


TWO_USERS = """
[[steps]]
name = "plan"

[[steps]]
name = "build"
[steps.agent]
user = "builder"

[[steps]]
name = "check"
[steps.agent]
user = 1000
"""


async def test_each_turn_runs_as_its_steps_agent_user(tmp_path, monkeypatch):
    """Harbor runs each step as its own `[steps.agent].user` (else the task's `[agent].user`): the sandbox's
    default user, the working directory's owner and the user the harness is handed switch with the turn, once,
    when the step's user differs from the one in place; a numeric uid is resolved to its name as at seed, and a
    step without a user runs as the task's. The verifier keeps its own fallback (the task image's user)."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        TWO_USERS,
        {"plan": "P", "build": "B", "check": "C"},
        agent_implementation="harbor_oracle_agent",
    )
    write_step_solution(task.path, "plan", "#!/bin/bash\necho plan\n")
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    users_at_turn: list = []
    server.server_client.turn_hooks.append(lambda: users_at_turn.append(sandbox.default_user))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build", "check"]
    # The task's user is root (no [agent].user, a root image): the first turn keeps the image's default.
    assert users_at_turn == [None, "builder", "agent"]
    assert [seed["sandbox_access"].get("user") for seed in server.server_client.seed_bodies()] == [
        None,
        "builder",
        "agent",
    ]
    # The working directory is handed to each new user as root, after the seed's own mkdir.
    chowns = [c for c in sandbox.execs if c["command"].startswith("chown ")]
    assert [(c["command"], c["user"]) for c in chowns] == [
        ("chown builder /app", "root"),
        ("chown agent /app", "root"),
    ]
    # test.sh never runs as the step's agent user: the verifier keeps its own user (the image's, here plain root,
    # which is the facade's `None` path), not `builder` or the uid.
    runs = [c for c in sandbox.execs if "bash /tests/test.sh" in c["command"]]
    assert [run["user"] for run in runs] == [None, None, None]
    # The uid was looked up once, by name, as root.
    lookups = [c for c in sandbox.execs if c["command"].startswith("getent passwd 1000")]
    assert len(lookups) == 1
    # A step solution placed for the first turn was uploaded before any switch; the switch leaves it readable.
    assert uploaded_files(sandbox, "solve.sh") == [b"#!/bin/bash\necho plan\n"]


async def test_consecutive_steps_with_the_same_user_switch_once_and_a_later_step_switches_back(tmp_path, monkeypatch):
    steps = (
        TASK_TOML.replace("[agent]\n", '[agent]\nuser = "agent"\n')
        + '[[steps]]\nname = "a"\n[steps.agent]\nuser = "builder"\n'
        + '[[steps]]\nname = "b"\n[steps.agent]\nuser = "builder"\n'
        + '[[steps]]\nname = "c"\n'
    )
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"a": "A", "b": "B", "c": "C"})
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=steps)
    assert task.user == "agent"
    users_at_turn: list = []
    server.server_client.turn_hooks.append(lambda: users_at_turn.append(sandbox.default_user))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert users_at_turn == ["builder", "builder", "agent"]
    chowns = [c["command"] for c in sandbox.execs if c["command"].startswith("chown ") or "&& chown " in c["command"]]
    # The seed's own hand-over to the task user, then one per switch: to builder before `a`, back to agent before `c`.
    assert chowns == ["mkdir -p /app && chown agent /app", "chown builder /app", "chown agent /app"]


@pytest.mark.parametrize("provider", ["docker", "opensandbox"])
async def test_a_step_agent_user_with_a_group_follows_the_provider_at_seed(tmp_path, monkeypatch, provider):
    """A step's `[steps.agent].user = "name:group"` is as static as the task's own: Docker's `--user` takes the
    pair, every other provider switches user by name only, so the pair is refused at seed there, naming the step,
    before any turn is paid for (the rule M16/M20 set for the task's `[agent].user`)."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[steps.agent]\nuser = "builder:grp"\n'
    server, task, sandbox, created = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {provider: {}}}
    )

    response = await server.run_request(request_for(task))

    if provider == "docker":
        assert response.failure is None, response.failure
        assert [turn.name for turn in response.result.turns] == ["plan", "build"]
        assert sandbox.default_user == "builder:grp"
        assert server.server_client.seed_bodies()[1]["sandbox_access"]["user"] == "builder:grp"
    else:
        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "hello step 'build': user 'builder:grp' names group 'grp'" in failure.failure_reason
        assert created == [] and sandbox.execs == [] and server.server_client.seed_bodies() == []


LONG_STEPS = """
[[steps]]
name = "plan"
[steps.agent]
timeout_sec = 0.3
[steps.verifier]
timeout_sec = 0.3

[[steps]]
name = "build"
[steps.agent]
timeout_sec = 0.3
[steps.verifier]
timeout_sec = 0.3
"""


async def test_a_multi_step_task_longer_than_the_default_gets_its_own_budget(tmp_path, monkeypatch, caplog):
    """G7 round 24: MemoryArena's six steps budget 21 780 s, above `default_episode_timeout_seconds` (21 600), and
    the seed only warned, so a real run would be cut off at the default and retried, never scored. With
    `enforce_task_timeouts` the episode deadline and the sandbox lifetime now follow the task's own budget when it
    is the longer one, the way the single-turn path already derives the deadline from the task."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        LONG_STEPS,
        {"plan": "P", "build": "B"},
        default_episode_timeout_seconds=0.1,
        harness_setup_allowance_s=0,
        verifier_grace_s=0,
        sandbox_ready_timeout_s=1,
    )
    # Two turns of 0.3 + 0.3 s plus one sandbox start: well above the run-wide default of 0.1 s.
    assert server._turns_budget_s(task) == pytest.approx(1.2)
    assert server._task_budget_s(task) == pytest.approx(2.2)
    assert server._episode_budget_s(task) == pytest.approx(2.2)
    assert await server.episode_timeout_seconds(request_for(task)) == pytest.approx(2.2)
    # A single-turn task keeps the run-wide cap.
    single_server, single_task, _, _ = make_server(
        tmp_path / "single", monkeypatch, default_episode_timeout_seconds=0.1
    )
    assert single_server._episode_budget_s(single_task) == 0.1
    assert await single_server.episode_timeout_seconds(request_for(single_task)) == 0.1
    # The harness takes 0.5 s per turn: over the default, within the task's budget, so both turns are scored.
    server.server_client.slow_path = "/ng-rollout/r1/v1/responses"
    with caplog.at_level("WARNING"):
        response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build"] and response.result.reward == 1.0
    assert (
        "above default_episode_timeout_seconds=0; the episode deadline and the sandbox lifetime follow" in caplog.text
    )
    # The derived deadline is the task's own budget, not a blank cheque: `test_the_task_deadline_ends_an_overrunning_turn`
    # keeps the task deadline firing on a turn that runs past it.


async def test_without_enforcement_a_multi_step_task_longer_than_the_default_is_refused_naming_the_setting(
    tmp_path, monkeypatch
):
    server, task, sandbox, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        LONG_STEPS,
        {"plan": "P", "build": "B"},
        default_episode_timeout_seconds=0.1,
        verifier_grace_s=0,
        enforce_task_timeouts=False,
    )
    assert server._episode_budget_s(task) == 0.1 and await server.episode_timeout_seconds(request_for(task)) == 0.1

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure is not None and failure.stage == "seed" and failure.terminal
    assert "its 2 steps budget up to 1s" in failure.failure_reason
    assert (
        "default_episode_timeout_seconds=0" in failure.failure_reason
        and "enforce_task_timeouts" in failure.failure_reason
    )
    assert "raise default_episode_timeout_seconds to at least 1" in failure.failure_reason
    assert created == [] and sandbox.execs == [] and server.server_client.seed_bodies() == []
    # A default that covers the steps runs them without enforcement, as before.
    server.config.default_episode_timeout_seconds = 5
    response = await server.run_request(request_for(task, rollout="r2"))
    assert response.failure is None and [turn.name for turn in response.result.turns] == ["plan", "build"]


async def test_a_forced_separate_verifier_is_refused_at_seed_for_a_multi_step_task(tmp_path, monkeypatch):
    """The loader admits multi-step tasks in shared mode only; `separate_verifier_for_artifacts` cannot force
    them into a mode their turns do not run, and the refusal costs no sandbox."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        steps,
        {"plan": "P", "build": "B"},
        head='artifacts = ["/app/out.txt"]\n',
        separate_verifier_for_artifacts=True,
    )

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure.stage == "seed" and failure.terminal
    assert "multi-step task" in failure.failure_reason and "separate verification" in failure.failure_reason
    assert not created


MCP_TOML = """
[[environment.mcp_servers]]
name = "user-server"
transport = "streamable-http"
url = "http://user-server:8000/mcp"

[[environment.mcp_servers]]
name = "legacy"
transport = "sse"
url = "http://legacy:9000/sse"
"""


@pytest.mark.parametrize("grants", ["off", "optional", "required"])
async def test_task_mcp_servers_are_granted_on_request(tmp_path, monkeypatch, grants):
    server, task, _, _ = make_server(tmp_path, monkeypatch, toml=TASK_TOML + MCP_TOML, task_mcp_grants=grants)

    await server.run_request(request_for(task))

    (seed,) = server.server_client.seed_bodies()
    if grants == "off":
        assert seed["tool_accesses"] == []
    else:
        # Only the streamable-http entry can be granted; the URL is the sidecar's own host, as declared.
        assert seed["tool_accesses"] == [
            {
                "kind": "mcp",
                "name": "user-server",
                "required": grants == "required",
                "connection": {"transport": "streamable_http", "url": "http://user-server:8000/mcp", "headers": {}},
            }
        ]


def turn(
    name: str,
    rewards: dict[str, float] | None,
    *,
    masked: bool = False,
    components: dict[str, float] | None = None,
    reward: float | None = None,
) -> HarborTurnResult:
    if reward is None:
        reward = (rewards or {}).get("reward", 0.0)
    return HarborTurnResult(
        name=name,
        reward=reward,
        verifier_rewards=rewards,
        reward_components=components,
        mask_sample=masked,
        failure_kind="provider_unavailable" if masked else None if rewards else "harbor:missing_reward",
        failure_reason="lost" if masked else None if rewards else "no reward.json or reward.txt was written",
        response=agent_response(),
    )


class TestRollUp:
    def test_a_masked_turn_masks_the_episode(self):
        rolled = roll_up_turns([turn("a", {"reward": 1.0}), turn("b", None, masked=True)], "mean")
        assert rolled["mask_sample"] and rolled["failure_kind"] == "provider_unavailable"

    def test_mean_counts_a_missing_key_as_zero(self):
        rolled = roll_up_turns([turn("a", {"reward": 1.0, "x": 1.0}), turn("b", {"reward": 0.0})], "mean")
        assert rolled == {"reward": 0.5, "verifier_rewards": {"reward": 0.5, "x": 0.5}}

    def test_mean_without_any_result_is_measured_zero_with_the_turns_cause(self):
        rolled = roll_up_turns([turn("a", None)], "mean")
        assert (rolled["reward"], rolled["mask_sample"], rolled["failure_kind"]) == (
            0.0,
            False,
            "harbor:missing_reward",
        )
        unexplained = turn("a", None).model_copy(update={"failure_kind": None})
        assert roll_up_turns([unexplained], "mean")["failure_kind"] == NO_TURN_RESULT_KIND

    def test_final_takes_the_last_turn_that_ran(self):
        assert roll_up_turns([turn("a", {"reward": 0.0}), turn("b", {"reward": 1.0})], "final")["reward"] == 1.0
        assert roll_up_turns([turn("a", {"reward": 1.0}), turn("b", None)], "final")["reward"] == 0.0

    def test_final_keeps_the_last_turns_components(self, caplog):
        parts = {"x": 0.5, "y": 0.5}
        last = turn("b", parts, components=parts, reward=1.0)
        with caplog.at_level(logging.WARNING, logger="environment_servers.harbor.app"):
            rolled = roll_up_turns([turn("a", {"reward": 0.0}), last], "final", task_id="t")
        assert rolled == {"reward": 1.0, "verifier_rewards": parts, "reward_components": parts}
        assert not caplog.records
        # Components that do not sum to the reward are never named, and the drop is said, not silent: the
        # warning names the task, the keys, their sum and the reward.
        off = turn("b", parts, components={"x": 0.5}, reward=1.0)
        with caplog.at_level(logging.WARNING, logger="environment_servers.harbor.app"):
            assert "reward_components" not in roll_up_turns([off], "final", task_id="t")
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert "task 't'" in message and "['x']" in message and "turn 'b'" in message
        assert "sum to 0.5, not to the episode reward 1.0" in message

    def test_mean_names_components_only_when_every_turn_did(self):
        first = turn("a", {"x": 1.0}, components={"x": 1.0}, reward=1.0)
        second = turn("b", {"x": 0.0, "y": 0.5}, components={"x": 0.0, "y": 0.5}, reward=0.5)
        rolled = roll_up_turns([first, second], "mean")
        assert rolled["reward"] == 0.75 and rolled["reward_components"] == {"x": 0.5, "y": 0.25}
        # A turn that named its reward directly has no components, so the episode names none either.
        mixed = roll_up_turns([first, turn("b", {"reward": 0.5})], "mean")
        assert mixed["reward"] == 0.75 and "reward_components" not in mixed

    def test_mean_drops_components_that_do_not_sum_with_a_warning(self, caplog):
        first = turn("a", {"x": 1.0}, components={"x": 1.0}, reward=1.0)
        # A turn whose components do not sum to its reward (nothing on the verifier path builds one) pulls the
        # per-key means off the mean reward; the episode then names none and says so.
        off = turn("b", {"x": 0.0, "y": 0.5}, components={"x": 0.0}, reward=0.5)
        with caplog.at_level(logging.WARNING, logger="environment_servers.harbor.app"):
            rolled = roll_up_turns([first, off], "mean", task_id="t")
        assert rolled["reward"] == 0.75 and "reward_components" not in rolled
        [record] = caplog.records
        assert "task 't'" in record.getMessage() and "the mean over turns" in record.getMessage()
        assert "sum to 0.5, not to the episode reward 0.75" in record.getMessage()


@pytest.mark.parametrize(
    ("rewards", "min_reward", "fails"),
    [
        ({"reward": 1.0}, None, False),
        ({"reward": 0.5}, 0.5, False),
        ({"reward": 0.4}, 0.5, True),
        ({"reward": 1.0, "tests": 0.0}, {"tests": 1.0}, True),
        ({"reward": 1.0}, {"tests": 0.0}, True),
        (None, 0.0, True),
    ],
)
def test_min_reward_follows_harbor(rewards, min_reward, fails):
    assert (min_reward_failure(rewards, min_reward) is not None) == fails
