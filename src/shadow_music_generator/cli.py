from __future__ import annotations

import argparse
import json

from .jobs import GenerationRequest, GenerationJob, JobStore


def job_payload(job: GenerationJob) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "mode": job.request.mode,
        "request": job.request.__dict__,
        "model_license": job.model_license,
        "pipeline": job.pipeline,
        "stages": [
            {"name": stage.name, "status": stage.status, "duration_ms": stage.duration_ms, "detail": stage.detail}
            for stage in job.stages
        ],
        "outputs": list(job.outputs),
        "output": job.output,
        "error": job.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="shadow-music-generator",
        description="Queue and run isolated music generation jobs (ShadowRoom Music). Dry-run never touches a model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="Queue a generation job.")
    submit.add_argument("--prompt", required=True)
    submit.add_argument("--mode", choices=["dry-run", "local"], default="dry-run")
    submit.add_argument("--model")
    submit.add_argument("--lyrics")
    submit.add_argument("--source-audio")
    submit.add_argument("--output-dir")
    submit.add_argument("--job-dir")
    submit.add_argument("--factory", help="Override SHADOW_PIPELINE_FACTORY for this job.")
    submit.add_argument("--run", action="store_true", help="Run the job immediately instead of leaving it queued.")
    run = commands.add_parser("run", help="Run a queued job.")
    run.add_argument("--job", required=True)
    run.add_argument("--job-dir")
    run.add_argument("--factory", help="Override SHADOW_PIPELINE_FACTORY for this job.")
    status = commands.add_parser("status", help="Show one job.")
    status.add_argument("--job", required=True)
    status.add_argument("--job-dir")
    listing = commands.add_parser("list", help="List every known job.")
    listing.add_argument("--job-dir")
    args = parser.parse_args()

    store = JobStore(args.job_dir)
    if args.command == "submit":
        job = store.submit(
            GenerationRequest(args.prompt, args.mode, args.model, args.output_dir, args.lyrics, args.source_audio)
        )
        if job.request.mode == "dry-run" or args.run:
            job = store.run(job.id, args.factory)
    elif args.command == "run":
        job = store.run(args.job, args.factory)
    elif args.command == "list":
        print(json.dumps([job_payload(item) for item in store.jobs()], ensure_ascii=False, indent=2))
        return 0
    else:
        job = store.get(args.job)
    print(json.dumps(job_payload(job), ensure_ascii=False, indent=2))
    return 1 if job.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
