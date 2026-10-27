"""Ask Codex directly: run a prompt pack through ``codex exec`` and collect the art.

Claude writes the prompts; Codex paints. Each prompt becomes one non-interactive
``codex exec`` job in the art folder:

    codex exec --skip-git-repo-check --enable image_generation -s workspace-write
               -C <art> --ephemeral -o <log>.last.txt [-m MODEL] -i ref1.png -i ref2.png   < prompt

The prompt goes through stdin, so no shell quoting can mangle it. Jobs run in
dependency order (design -> turnaround -> exploded views and head sheet) with at
most ``parallel`` jobs at once (3 by default). Every exploded view is checked with
the rig's own segmentation; a failing view is sent back to Codex with the
corrective prompt and the rejected drawing attached, up to ``retries`` times.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import image as im
from . import prompts as pr

DEPENDS = {"design": [], "turnaround": ["design"], "exploded": ["design", "turnaround"], "heads": ["design"]}


@dataclass
class Job:
    name: str  # output file stem, unique
    step: str
    prompt: str
    output: str  # file name inside the art folder
    attach: list[str]
    after: list[str]  # job names that must succeed first
    views: list[str] = field(default_factory=list)  # views drawn in this image (exploded only)


@dataclass
class JobResult:
    name: str
    status: str = "pending"  # ok | failed | skipped
    attempts: int = 0
    findings: list[str] = field(default_factory=list)
    message: str = ""


# runner(job, command, prompt, cwd, log_path) -> exit code
Runner = Callable[[Job, list[str], str, Path, Path], int]


def wrap(job: Job) -> str:
    refs = ", ".join(job.attach) if job.attach else "none"
    return (
        "You are the art department for a 2D pixel game. Generate ONE image with your image generation tool and "
        f'save it as "{job.output}" (PNG) in the current working directory, replacing any existing file of that name. '
        f"Attached reference images: {refs}. Do not edit, move or delete any other file. "
        f"When done, reply with the file name only.\n\nImage brief:\n{job.prompt}\n"
    )


def jobs_from_pack(pack: dict) -> list[Job]:
    """Turn ``prompts.json`` (from ``spritekit prompts --out``) into jobs."""
    by_step: dict[str, list[str]] = {}
    jobs = []
    for p in pack["prompts"]:
        name = Path(p["file"]).stem
        by_step.setdefault(p["step"], []).append(name)
        views = []
        if p["step"] == "exploded":
            views = pack["views"] if pack["layout"] == "sheet" else [name.removeprefix("view_")]
        jobs.append(Job(name, p["step"], p["text"], p["file"], list(p["attach"]), [], views))
    for j in jobs:
        j.after = [n for dep in DEPENDS.get(j.step, []) for n in by_step.get(dep, [])]
    return jobs


def codex_command(job: Job, art: Path, codex: str = "codex", model: str | None = None) -> list[str]:
    cmd = [codex, "exec", "--skip-git-repo-check", "--enable", "image_generation", "-s", "workspace-write",
           "-C", str(art), "--ephemeral", "-o", str(art / "codex_logs" / f"{job.name}.last.txt")]
    if model:
        cmd += ["-m", model]
    for a in job.attach:
        cmd += ["-i", str(art / a)]
    return cmd


def subprocess_runner(job: Job, cmd: list[str], prompt: str, cwd: Path, log: Path) -> int:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, input=prompt, text=True, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


def _check(job: Job, art: Path) -> tuple[bool, list[str], str | None]:
    """Is the output present, loadable and (for exploded views) rigable?"""
    path = art / job.output
    if not path.exists():
        return False, [f"Codex did not save {job.output}"], None
    try:
        drawing = im.load_rgba(path)
    except OSError as e:
        return False, [f"{job.output} is not a readable image: {e}"], None
    if job.step != "exploded":
        return True, [], None
    review = pr.review(drawing, job.views)
    return review.ok, [f"[{f.figure}] {f.problem}" for f in review.findings], review.fix_prompt


def run(pack: dict, art: str | Path, codex: str = "codex", model: str | None = None, parallel: int = 3,
        retries: int = 2, only: set[str] | None = None, runner: Runner = subprocess_runner,
        log: Callable[[str], None] = print) -> dict[str, JobResult]:
    """Run every job, at most ``parallel`` at a time, retrying failing exploded views."""
    art = Path(art)
    (art / "codex_logs").mkdir(parents=True, exist_ok=True)
    jobs = {j.name: j for j in jobs_from_pack(pack)}
    results = {n: JobResult(n) for n in jobs}
    pending = [n for n, j in jobs.items() if only is None or j.step in only]
    for n, j in jobs.items():
        if n not in pending:  # not requested: usable only if its image is already there
            results[n].status = "ok" if (art / j.output).exists() else "skipped"
            if results[n].status == "skipped":
                results[n].message = "not requested and no existing image"

    def attempt(job: Job, prompt_job: Job) -> int:
        results[job.name].attempts += 1
        cmd = codex_command(prompt_job, art, codex, model)
        return runner(prompt_job, cmd, wrap(prompt_job), art, art / "codex_logs" / f"{job.name}.log")

    running: dict[Future, tuple[Job, float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        while pending or running:
            for name in list(pending):
                job = jobs[name]
                states = [results[d].status for d in job.after]
                if any(s in ("failed", "skipped") for s in states):
                    pending.remove(name)
                    results[name].status = "skipped"
                    results[name].message = "a reference image it needs failed"
                    log(f"skip {name}: a reference image it needs failed")
                elif all(s == "ok" for s in states) and len(running) < parallel:
                    pending.remove(name)
                    log(f"codex: {name} (attempt 1)")
                    running[pool.submit(attempt, job, job)] = (job, time.monotonic())
            if not running:
                if pending:  # nothing can start and nothing is running: dependencies never resolved
                    for name in pending:
                        results[name].status = "skipped"
                        results[name].message = "dependencies never resolved"
                    pending.clear()
                continue
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                job, started = running.pop(fut)
                res = results[job.name]
                try:
                    code = fut.result()
                except OSError as e:  # e.g. codex not installed
                    code, res.message = 127, str(e)
                ok, findings, fix = _check(job, art) if code == 0 else (False, [f"codex exited with {code}"], None)
                res.findings = findings
                took = time.monotonic() - started
                if ok:
                    res.status = "ok"
                    log(f"  ok   {job.output} ({res.attempts} attempt(s), {took:.0f}s)")
                elif res.attempts <= retries and code == 0:
                    log(f"  redo {job.output}: {'; '.join(findings)}")
                    if (art / job.output).exists():
                        shutil.copy(art / job.output, art / f"{job.name}.try{res.attempts}.png")
                    if fix:  # send the rejected drawing back with the corrections
                        redo = Job(job.name, job.step, f"{fix}\n\nOriginal brief, for reference:\n{job.prompt}",
                                   job.output, [job.output] + job.attach, job.after, job.views)
                    else:  # nothing usable came back: ask again with the original brief
                        redo = job
                    running[pool.submit(attempt, job, redo)] = (job, time.monotonic())
                else:
                    res.status = "failed"
                    res.message = res.message or "; ".join(findings)
                    log(f"  FAIL {job.output}: {res.message}")
    (art / "codex_run.json").write_text(json.dumps({n: asdict(r) for n, r in results.items()}, indent=1))
    return results


# ---------------------------------------------------------------------------
# PowerShell script, Avaruun style (art/codex/*.ps1, Start-Job, 3 at a time)


def powershell_script(pack: dict, codex: str = "codex", model: str | None = None, parallel: int = 3) -> str:
    """A self-contained .ps1 that runs the same jobs; prompts are read from files, never inlined."""
    jobs = jobs_from_pack(pack)
    waves: list[list[Job]] = []
    done: set[str] = set()
    rest = list(jobs)
    while rest:
        wave = [j for j in rest if set(j.after) <= done]
        waves.append(wave)
        done |= {j.name for j in wave}
        rest = [j for j in rest if j not in wave]
    lines = [
        "# Generated by spritekit: ask Codex for every image in this prompt pack.",
        "# Run from anywhere: powershell -ExecutionPolicy Bypass -File run_codex.ps1",
        "$ErrorActionPreference = 'Stop'",
        "$Art = $PSScriptRoot",
        f"$Codex = '{codex}'",
        f"$Throttle = {parallel}",
        "New-Item -ItemType Directory -Force -Path (Join-Path $Art 'codex_logs') | Out-Null",
        "function Start-CodexJob([string]$Name, [string[]]$Attach) {",
        "  Start-Job -Name $Name -ArgumentList $Art, $Codex, $Name, $Attach -ScriptBlock {",
        "    param($Art, $Codex, $Name, $Attach)",
        "    $OutputEncoding = New-Object System.Text.UTF8Encoding $false  # Windows PowerShell pipes ASCII otherwise",
        "    $prompt = Get-Content -Raw -Encoding UTF8 (Join-Path $Art \"codex_prompts\\$Name.txt\")",
        "    $cargs = @('exec', '--skip-git-repo-check', '--enable', 'image_generation', '-s', 'workspace-write',",
        "              '-C', $Art, '--ephemeral', '-o', (Join-Path $Art \"codex_logs\\$Name.last.txt\"))",
    ]
    if model:
        lines.append(f"    $cargs += @('-m', '{model}')")
    lines += [
        "    foreach ($a in $Attach) { $cargs += @('-i', (Join-Path $Art $a)) }",
        "    $prompt | & $Codex @cargs *> (Join-Path $Art \"codex_logs\\$Name.log\")",
        "  } | Out-Null",
        "}",
        "function Wait-Throttle { while (@(Get-Job -State Running).Count -ge $Throttle) { Start-Sleep -Seconds 5 } }",
    ]
    for i, wave in enumerate(waves, 1):
        lines.append(f"# wave {i}: {', '.join(j.name for j in wave)}")
        for j in wave:
            attach = ", ".join(f"'{a}'" for a in j.attach) or ""
            lines.append(f"Wait-Throttle; Start-CodexJob '{j.name}' @({attach}); Write-Host 'codex: {j.output}'")
        lines.append("Get-Job | Wait-Job | Out-Null; Get-Job | Remove-Job")
        for j in wave:
            lines.append(f"if (-not (Test-Path (Join-Path $Art '{j.output}'))) "
                         f"{{ throw 'Codex did not save {j.output}; see codex_logs\\{j.name}.log' }}")
    lines.append("Write-Host 'All images saved. Next: python -m spritekit check ... (see prompts.md)'")
    return "\n".join(lines) + "\n"


def write_prompt_files(pack: dict, art: Path) -> None:
    """codex_prompts/<job>.txt holds each wrapped prompt for the PowerShell script."""
    d = art / "codex_prompts"
    d.mkdir(parents=True, exist_ok=True)
    for j in jobs_from_pack(pack):
        (d / f"{j.name}.txt").write_text(wrap(j), encoding="utf-8")
