#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://10.2.24.10:30890"
DEFAULT_IMAGES = [
    "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_kenvue_jp_yolov5-20251130",
    "ccr.ccs.tencentyun.com/clobotics/sku-model-init:sku_kenvue_jp_resnet-20260529",
    "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_suntory_yolov5-20260320",
    "ccr.ccs.tencentyun.com/clobotics/sku-model-init:sku_suntory_sg_resnet-20260525",
    "ccr.ccs.tencentyun.com/clobotics/sku-model-init:sku_suntory_tw_resnet-20260529",
    "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hpc_us_yolov8-20260107",
    "ccr.ccs.tencentyun.com/clobotics/sku-model-init:sku_hpc_us_resnet-20260604",
    "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620",
    "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
]

IMAGE_TAG_RELEASE_SUFFIX_PATTERN = re.compile(r"[-_.]?\d{6,}$")
MODEL_NAME_DERIVE_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_]+")


class BenchmarkError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton hot load/unload via HTTP API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Hot loader base URL.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="How many rounds to run for each unique image.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for job/status checks.",
    )
    parser.add_argument(
        "--load-timeout",
        type=float,
        default=1800.0,
        help="Timeout in seconds for a load job to reach terminal state.",
    )
    parser.add_argument(
        "--unload-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for unload to make the model non-READY.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full benchmark result JSON.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        help="Optional explicit image list. Defaults to the unique images provided in the request.",
    )
    return parser.parse_args()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def derive_model_name(image_ref: str) -> str:
    ref_without_digest = image_ref.split("@", 1)[0]
    last_slash_index = ref_without_digest.rfind("/")
    last_colon_index = ref_without_digest.rfind(":")
    if last_colon_index > last_slash_index:
        candidate = ref_without_digest[last_colon_index + 1 :]
        candidate = IMAGE_TAG_RELEASE_SUFFIX_PATTERN.sub("", candidate.strip())
    else:
        candidate = ref_without_digest[last_slash_index + 1 :].strip()
    candidate = candidate.lower().replace("-", "_").replace(".", "_")
    candidate = MODEL_NAME_DERIVE_SANITIZE_PATTERN.sub("_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        raise BenchmarkError(f"Cannot derive model name from image: {image_ref}")
    return candidate


def request_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 60.0) -> Any:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"{method.upper()} {path} failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"{method.upper()} {path} failed: {exc}") from exc
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{method.upper()} {path} returned non-JSON body: {body[:400]}") from exc


def get_status(base_url: str) -> dict[str, Any]:
    payload = request_json(base_url, "GET", "/api/status")
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Unexpected /api/status payload: {payload!r}")
    return payload


def get_job(base_url: str, job_name: str) -> dict[str, Any]:
    payload = request_json(base_url, "GET", f"/api/jobs/{urllib.parse.quote(job_name)}")
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Unexpected /api/jobs payload: {payload!r}")
    return payload


def find_model_entries(status_payload: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    triton_payload = status_payload.get("triton", {})
    if not isinstance(triton_payload, dict):
        return []
    repository_models = triton_payload.get("repository_models", [])
    if not isinstance(repository_models, list):
        return []
    return [
        item
        for item in repository_models
        if isinstance(item, dict) and str(item.get("name") or "") == model_name
    ]


def summarize_model_state(status_payload: dict[str, Any], model_name: str) -> str:
    entries = find_model_entries(status_payload, model_name)
    if not entries:
        return "ABSENT"
    states = sorted({str(item.get("state") or "UNKNOWN").upper() for item in entries})
    return ",".join(states) if states else "UNKNOWN"


def model_is_ready(status_payload: dict[str, Any], model_name: str) -> bool:
    return any(str(item.get("state") or "").upper() == "READY" for item in find_model_entries(status_payload, model_name))


def wait_for_job_terminal_state(
    base_url: str,
    job_name: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], float]:
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    while True:
        payload = get_job(base_url, job_name)
        status = str(payload.get("status") or "").upper()
        if status in {"MODEL_READY", "COPY_FAILED", "TRITON_RELOAD_FAILED"}:
            return payload, time.monotonic() - started
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                f"Timed out waiting for job {job_name} terminal state, last status={status or '-'} detail={payload.get('detail')}"
            )
        time.sleep(max(poll_interval_seconds, 0.1))


def wait_for_model_non_ready(
    base_url: str,
    model_name: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], float]:
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    while True:
        payload = get_status(base_url)
        if not model_is_ready(payload, model_name):
            return payload, time.monotonic() - started
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                f"Timed out waiting for model to become non-READY: {model_name}, last_state={summarize_model_state(payload, model_name)}"
            )
        time.sleep(max(poll_interval_seconds, 0.1))


def ensure_model_unloaded(
    base_url: str,
    model_name: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    status_payload = get_status(base_url)
    if not model_is_ready(status_payload, model_name):
        return {
            "cleanup_needed": False,
            "cleanup_success": True,
            "state_before": summarize_model_state(status_payload, model_name),
            "seconds": 0.0,
        }
    request_json(base_url, "POST", "/api/models/unload", {"model_name": model_name})
    final_status, seconds = wait_for_model_non_ready(
        base_url,
        model_name,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    return {
        "cleanup_needed": True,
        "cleanup_success": True,
        "state_before": "READY",
        "state_after": summarize_model_state(final_status, model_name),
        "seconds": seconds,
    }


def run_cycle(
    *,
    base_url: str,
    image: str,
    iteration: int,
    poll_interval_seconds: float,
    load_timeout_seconds: float,
    unload_timeout_seconds: float,
) -> dict[str, Any]:
    model_name = derive_model_name(image)
    result: dict[str, Any] = {
        "image": image,
        "model_name": model_name,
        "iteration": iteration,
        "load_success": False,
        "unload_success": False,
        "cleanup": None,
    }

    result["cleanup"] = ensure_model_unloaded(
        base_url,
        model_name,
        timeout_seconds=unload_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    load_submit_started = time.monotonic()
    load_response = request_json(base_url, "POST", "/api/models/load", {"image": image})
    result["load_submit_seconds"] = time.monotonic() - load_submit_started
    result["load_response"] = load_response
    result["job_name"] = str(load_response.get("job_name") or "")

    if not result["job_name"]:
        raise BenchmarkError(f"Load response missing job_name: {load_response}")

    job_payload, load_seconds = wait_for_job_terminal_state(
        base_url,
        result["job_name"],
        timeout_seconds=load_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    load_status = str(job_payload.get("status") or "").upper()
    result["load_seconds"] = load_seconds
    result["load_status"] = load_status
    result["load_detail"] = job_payload.get("detail")
    result["load_success"] = load_status == "MODEL_READY"
    result["job_terminal_payload"] = job_payload

    status_after_load = get_status(base_url)
    result["state_after_load"] = summarize_model_state(status_after_load, model_name)

    if not result["load_success"]:
        try:
            cleanup_after_failure = ensure_model_unloaded(
                base_url,
                model_name,
                timeout_seconds=unload_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            result["post_failure_cleanup_error"] = str(exc)
        else:
            result["post_failure_cleanup"] = cleanup_after_failure
        return result

    unload_submit_started = time.monotonic()
    unload_response = request_json(base_url, "POST", "/api/models/unload", {"model_name": model_name})
    result["unload_submit_seconds"] = time.monotonic() - unload_submit_started
    result["unload_response"] = unload_response

    status_after_unload, unload_seconds = wait_for_model_non_ready(
        base_url,
        model_name,
        timeout_seconds=unload_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    result["unload_seconds"] = unload_seconds
    result["state_after_unload"] = summarize_model_state(status_after_unload, model_name)
    result["unload_success"] = True
    return result


def average(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    load_attempts = len(results)
    load_successes = [item for item in results if item.get("load_success")]
    unload_attempts = [item for item in results if item.get("load_success")]
    unload_successes = [item for item in results if item.get("unload_success")]

    per_model: dict[str, dict[str, Any]] = {}
    for item in results:
        model_name = str(item["model_name"])
        entry = per_model.setdefault(
            model_name,
            {
                "image": item["image"],
                "attempts": 0,
                "load_successes": 0,
                "unload_successes": 0,
                "load_seconds_successes": [],
                "unload_seconds_successes": [],
                "load_failures": [],
                "unload_failures": [],
            },
        )
        entry["attempts"] += 1
        if item.get("load_success"):
            entry["load_successes"] += 1
            entry["load_seconds_successes"].append(item.get("load_seconds"))
            if item.get("unload_success"):
                entry["unload_successes"] += 1
                entry["unload_seconds_successes"].append(item.get("unload_seconds"))
            else:
                entry["unload_failures"].append(
                    {
                        "iteration": item["iteration"],
                        "error": item.get("error"),
                        "state_after_load": item.get("state_after_load"),
                    }
                )
        else:
            entry["load_failures"].append(
                {
                    "iteration": item["iteration"],
                    "status": item.get("load_status"),
                    "detail": item.get("load_detail"),
                }
            )

    for entry in per_model.values():
        entry["avg_load_seconds"] = average([value for value in entry.pop("load_seconds_successes") if isinstance(value, (int, float))])
        entry["avg_unload_seconds"] = average([value for value in entry.pop("unload_seconds_successes") if isinstance(value, (int, float))])
        entry["load_success_rate"] = entry["load_successes"] / entry["attempts"] if entry["attempts"] else None
        entry["unload_success_rate"] = (
            entry["unload_successes"] / entry["load_successes"] if entry["load_successes"] else None
        )

    return {
        "load_attempts": load_attempts,
        "load_success_count": len(load_successes),
        "load_success_rate": len(load_successes) / load_attempts if load_attempts else None,
        "avg_load_seconds": average(
            [item["load_seconds"] for item in load_successes if isinstance(item.get("load_seconds"), (int, float))]
        ),
        "unload_attempts": len(unload_attempts),
        "unload_success_count": len(unload_successes),
        "unload_success_rate": len(unload_successes) / len(unload_attempts) if unload_attempts else None,
        "avg_unload_seconds": average(
            [item["unload_seconds"] for item in unload_successes if isinstance(item.get("unload_seconds"), (int, float))]
        ),
        "per_model": per_model,
    }


def main() -> int:
    args = parse_args()
    images = dedupe_preserve_order(args.images or DEFAULT_IMAGES)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    preflight = get_status(args.base_url)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    print(f"Benchmark started at {started_at}")
    print(f"Base URL: {args.base_url}")
    print(f"Unique images: {len(images)}")
    print(f"Iterations per image: {args.iterations}")
    print(f"Triton ready: {preflight.get('triton', {}).get('ready')}")
    sys.stdout.flush()

    for iteration in range(1, args.iterations + 1):
        for index, image in enumerate(images, start=1):
            model_name = derive_model_name(image)
            print(f"[iter {iteration}/{args.iterations}] [{index}/{len(images)}] load/unload {model_name}")
            sys.stdout.flush()
            cycle_started = time.monotonic()
            try:
                cycle_result = run_cycle(
                    base_url=args.base_url,
                    image=image,
                    iteration=iteration,
                    poll_interval_seconds=args.poll_interval,
                    load_timeout_seconds=args.load_timeout,
                    unload_timeout_seconds=args.unload_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                cycle_result = {
                    "image": image,
                    "model_name": model_name,
                    "iteration": iteration,
                    "load_success": False,
                    "unload_success": False,
                    "error": str(exc),
                }
                failures.append(cycle_result)
                print(f"  FAILED: {exc}")
            else:
                print(
                    "  "
                    f"load={cycle_result.get('load_status')} "
                    f"load_s={cycle_result.get('load_seconds')} "
                    f"unload_ok={cycle_result.get('unload_success')} "
                    f"unload_s={cycle_result.get('unload_seconds')}"
                )
            cycle_result["cycle_seconds"] = time.monotonic() - cycle_started
            results.append(cycle_result)
            sys.stdout.flush()

    summary = build_summary(results)
    payload = {
        "started_at": started_at,
        "base_url": args.base_url,
        "iterations": args.iterations,
        "images": images,
        "preflight": preflight,
        "summary": summary,
        "results": results,
        "failures": failures,
        "definitions": {
            "load_success": "job terminal status == MODEL_READY",
            "unload_success": "after unload request, /api/status no longer reports the model as READY",
        },
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
