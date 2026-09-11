#!/usr/bin/env python3
"""Start/status/stop one owned detached overnight simulator job.

No sudo, system services, linger, robot processes or login settings are changed.
PID, boot ID and process start ticks protect stop against PID reuse. Detachment
requires logind KillUserProcesses=false; otherwise use an approved persistent
service instead. This tool does not change that host policy.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from artifacts import ROOT


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=pid, start_ticks=fields[19], boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except FileNotFoundError:
        return None


def live(record):
    return record.get("identity") is not None and process_identity(record["identity"]["pid"]) == record["identity"]


def jobs():
    result = []
    for path in sorted((ROOT/"runs").glob("*_overnight_job/job.json")):
        result.append((path, json.loads(path.read_text())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop"))
    parser.add_argument("--preflight-suite", type=Path)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--hours", type=float, default=8.)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-optimizer", action="store_true")
    args = parser.parse_args()
    with (ROOT/"runs/.overnight.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = jobs()
        active = [(p, r) for p, r in existing if live(r)]
        if args.command == "start":
            if active:
                raise ValueError("An owned overnight job is already running: "+str(active[-1][0]))
            if not args.preflight_suite or not args.preflight_suite.is_file():
                parser.error("start requires --preflight-suite suite.json")
            if not args.execute or not args.approve_optimizer:
                print("Not started. Explicit --execute and --approve-optimizer required.")
                return
            policy = subprocess.check_output(["busctl", "get-property", "org.freedesktop.login1",
                "/org/freedesktop/login1", "org.freedesktop.login1.Manager", "KillUserProcesses"], text=True).strip()
            if policy != "b false":
                raise ValueError("Host may kill jobs at logout; persistent service required")
            run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_overnight_job")
            run.mkdir(exist_ok=False)
            command = ["nohup", "nice", "-n", "5", sys.executable, str(ROOT/"m2_overnight.py"),
                "--preflight-suite", str(args.preflight_suite.resolve()), "--iterations", str(args.iterations),
                "--hours", str(args.hours), "--execute", "--approve-optimizer"]
            with (run/"supervisor.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(identity=process_identity(process.pid), command=command,
                          output=str(run/"supervisor.log"), simulation_only=True,
                          detached=True, host_logout_policy_changed=False)
            (run/"job.json").write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps(dict(job=str(run), pid=process.pid, started=True)))
            return
        if not existing:
            print("No overnight job recorded.")
            return
        path, record = active[-1] if active else existing[-1]
        if args.command == "stop":
            if live(record):
                os.kill(record["identity"]["pid"], signal.SIGINT)
                for _ in range(50):
                    if not live(record):
                        break
                    time.sleep(1)
                if live(record):
                    raise RuntimeError("Owned supervisor did not stop; inspect its logs before further action")
                print("Owned overnight job stopped. Existing checkpoints retained.")
            else:
                print("Overnight supervisor is not running; no signals sent.")
        print(json.dumps(dict(job=str(path.parent), running=live(record), log=record["output"])))
        log_path = Path(record["output"])
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            found = [x.removeprefix("Overnight supervisor: ") for x in lines if x.startswith("Overnight supervisor: ")]
            if found:
                status = Path(found[-1])/"status.json"
                if status.exists():
                    print(status.read_text())
            else:
                print("\n".join(lines[-10:]))


if __name__ == "__main__":
    main()
