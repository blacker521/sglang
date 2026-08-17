#!/usr/bin/env python3
"""用 EvalScope 压测 GenRec：TTFT / TPOT / E2E，并可选计算 SID 合法率。

前提：服务已启动；数据集为本目录下 genrec_perf_5000.jsonl。

  python benchmark/run_genrec_evalscope_perf.py \\
      --url http://127.0.0.1:8904/v1/chat/completions \\
      --parallel 1 8 --number 5000 5000 \\
      --valid-sid-file data/sid2vid.json --sid-valid-num 100
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    "/mnt/nas_new_1/mcu-multimodal-rec/model/online/mcu-genrec-v1.1.23-fp8/1784718542"
)
DEFAULT_DATASET = BENCH_DIR / "genrec_perf_5000.jsonl"
DEFAULT_VALID_SID_FILE = BENCH_DIR.parent / "data" / "sid2vid.json"

# GenRec beam output: <|sid_begin|><s_a_x><s_b_y><s_c_z><|sid_end|>
_SID_RE = re.compile(
    r"<\|sid_begin\|>\s*<s_a_(\d+)>\s*<s_b_(\d+)>\s*<s_c_(\d+)>\s*<\|sid_end\|>"
)


def _patch_evalscope_single_turn_cache_reporting() -> None:
    """Propagate server cache usage into EvalScope's single-turn aggregates.

    EvalScope parses ``usage.prompt_tokens_details.cached_tokens`` into
    ``real_cached_tokens``, but currently copies it to ``cached_tokens`` only in
    its multi-turn strategy. Workload throughput reads ``cached_tokens``, so
    otherwise a valid SGLang radix-cache hit is incorrectly reported as zero.
    """
    original = DefaultApiPlugin.process_request
    if getattr(original, "_sglang_cache_reporting_patch", False):
        return

    async def process_request_with_cache_reporting(self, *args, **kwargs):
        output = await original(self, *args, **kwargs)
        if output.cached_tokens is None and output.real_cached_tokens is not None:
            output.cached_tokens = output.real_cached_tokens
        return output

    process_request_with_cache_reporting._sglang_cache_reporting_patch = True
    DefaultApiPlugin.process_request = process_request_with_cache_reporting


def parse_int_list(values: Optional[Sequence[str]]) -> List[int]:
    out: List[int] = []
    for v in values or []:
        for part in str(v).replace(",", " ").split():
            if part:
                out.append(int(part))
    return out


def _load_dataset_rows(dataset_path: Path, limit: int) -> List[Any]:
    rows: List[Any] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _truncate(text: str, n: int = 160) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + "..."


def load_valid_sid_set(path: Path) -> Set[str]:
    """Load allowed SID keys from sid2vid / list JSON (keys like ``\"12,34,56\"``)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k).strip() for k in data if str(k).strip()}
    if isinstance(data, list):
        out: Set[str] = set()
        for item in data:
            if isinstance(item, str):
                key = item.strip()
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                key = ",".join(str(int(x)) for x in item[:3])
            else:
                continue
            if key:
                out.add(key)
        return out
    raise ValueError(f"{path}: expected sid2vid object or list of SID keys")


def parse_sid_key(content: Any) -> Optional[str]:
    """Extract ``a,b,c`` codebook key from a GenRec SID string; None if unparseable."""
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None
    m = _SID_RE.search(text)
    if not m:
        return None
    return f"{int(m.group(1))},{int(m.group(2))},{int(m.group(3))}"


def sid_validity_label(content: Any, valid_sids: Optional[Set[str]]) -> str:
    if valid_sids is None:
        return ""
    key = parse_sid_key(content)
    if key is None:
        return "parse_fail"
    return "valid" if key in valid_sids else "invalid"


def _post_chat_completion(
    *,
    url: str,
    model: str,
    messages: Any,
    n: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": n,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "separate_reasoning": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def compute_sid_validity(
    *,
    url: str,
    model: str,
    dataset_path: Path,
    n: int,
    max_tokens: int,
    temperature: float,
    num_requests: int,
    valid_sids: Set[str],
) -> Dict[str, Any]:
    """Query ``num_requests`` samples and measure SID-in-set legality rates."""
    rows = _load_dataset_rows(dataset_path, num_requests)
    total_beams = 0
    valid_beams = 0
    invalid_beams = 0
    parse_fail_beams = 0
    failed_requests = 0
    top1_valid = 0
    top1_total = 0
    invalid_examples: List[Dict[str, Any]] = []

    for i, messages in enumerate(rows, 1):
        try:
            data = _post_chat_completion(
                url=url,
                model=model,
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            failed_requests += 1
            print(f"[sid-valid] request {i} failed: {e!r}")
            continue

        choices = data.get("choices") or []
        req_top1_ok = False
        for j, c in enumerate(choices):
            content = (c.get("message") or {}).get("content")
            key = parse_sid_key(content)
            total_beams += 1
            if key is None:
                parse_fail_beams += 1
                label = "parse_fail"
            elif key in valid_sids:
                valid_beams += 1
                label = "valid"
                if j == 0:
                    req_top1_ok = True
            else:
                invalid_beams += 1
                label = "invalid"
                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        {"request": i, "beam": j, "sid": key, "content": content}
                    )
            if j == 0:
                top1_total += 1
                if req_top1_ok:
                    top1_valid += 1

        if i == 1 or i % 20 == 0 or i == len(rows):
            checked = valid_beams + invalid_beams
            rate = (valid_beams / checked) if checked else 0.0
            print(
                f"[sid-valid] progress {i}/{len(rows)} "
                f"beams={total_beams} valid_rate={rate:.4%} "
                f"(valid={valid_beams} invalid={invalid_beams} "
                f"parse_fail={parse_fail_beams})"
            )

    checked = valid_beams + invalid_beams
    beam_valid_rate = (valid_beams / checked) if checked else None
    beam_parse_ok_rate = (
        (checked / total_beams) if total_beams else None
    )
    top1_valid_rate = (top1_valid / top1_total) if top1_total else None

    return {
        "valid_sid_file_size": len(valid_sids),
        "num_requests": len(rows),
        "failed_requests": failed_requests,
        "total_beams": total_beams,
        "valid_beams": valid_beams,
        "invalid_beams": invalid_beams,
        "parse_fail_beams": parse_fail_beams,
        "beam_valid_rate": beam_valid_rate,
        "beam_parse_ok_rate": beam_parse_ok_rate,
        "top1_valid": top1_valid,
        "top1_total": top1_total,
        "top1_valid_rate": top1_valid_rate,
        "invalid_examples": invalid_examples,
    }


def print_sid_validity_report(stats: Dict[str, Any], out_path: Optional[Path] = None) -> None:
    def _pct(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.4%}"

    print("\n===== SID VALIDITY (合法率) =====")
    print(f"SID set size: {stats['valid_sid_file_size']}")
    print(
        f"requests: {stats['num_requests']} "
        f"(failed={stats['failed_requests']})"
    )
    print(
        f"beams: total={stats['total_beams']} "
        f"valid={stats['valid_beams']} "
        f"invalid={stats['invalid_beams']} "
        f"parse_fail={stats['parse_fail_beams']}"
    )
    print(f"beam_valid_rate (合法率): {_pct(stats['beam_valid_rate'])}")
    print(f"beam_parse_ok_rate: {_pct(stats['beam_parse_ok_rate'])}")
    print(
        f"top1_valid_rate: {_pct(stats['top1_valid_rate'])} "
        f"({stats['top1_valid']}/{stats['top1_total']})"
    )
    if stats.get("invalid_examples"):
        print("invalid examples (up to 10):")
        for ex in stats["invalid_examples"]:
            print(
                f"  req={ex['request']} beam={ex['beam']} "
                f"sid={ex['sid']} content={ex['content']!r}"
            )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[sid-valid] wrote {out_path}")
    print("===== END SID VALIDITY =====\n")


def _user_preview(messages: Any) -> str:
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(str(c.get("text", "")))
                        else:
                            parts.append(str(c))
                    return _truncate("".join(parts))
                return _truncate(str(content))
    return _truncate(str(messages))


def _inject_system_into_row(row: Any, system_prompt: str) -> Any:
    """Prepend a fixed system message to a line_by_line chat row."""
    if isinstance(row, list):
        msgs = [dict(m) if isinstance(m, dict) else m for m in row]
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            if msgs[0].get("content") == system_prompt:
                return msgs
            msgs[0] = {"role": "system", "content": system_prompt}
            return msgs
        return [{"role": "system", "content": system_prompt}, *msgs]
    if isinstance(row, dict):
        out = dict(row)
        if "messages" in out:
            out["messages"] = _inject_system_into_row(out["messages"], system_prompt)
            return out
        if "prompt" in out and isinstance(out["prompt"], list):
            out["prompt"] = _inject_system_into_row(out["prompt"], system_prompt)
            return out
    return row


def _write_dataset_with_shared_system(
    src: Path, system_prompt: str, outputs_dir: Path
) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dst = outputs_dir / f"{src.stem}_shared_system.jsonl"
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fout.write(
                json.dumps(
                    _inject_system_into_row(row, system_prompt), ensure_ascii=False
                )
                + "\n"
            )
    return dst


def print_sample_cases(
    *,
    url: str,
    model: str,
    dataset_path: Path,
    n: int,
    max_tokens: int,
    temperature: float,
    sample_num: int,
    top_k: int,
    valid_sids: Optional[Set[str]] = None,
) -> None:
    """压测开始前打若干条真实返回，确认格式与内容。"""
    if sample_num <= 0:
        return

    rows = _load_dataset_rows(dataset_path, sample_num)
    if not rows:
        print("[sample] dataset empty, skip")
        return

    print(f"\n[sample] printing {len(rows)} response case(s) before benchmark "
          f"(n={n}, max_tokens={max_tokens}, top_k={top_k})")

    for i, messages in enumerate(rows, 1):
        try:
            data = _post_chat_completion(
                url=url,
                model=model,
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"\n===== SAMPLE CASE {i} FAILED HTTP {e.code} =====")
            print(err_body[:500])
            continue
        except Exception as e:
            print(f"\n===== SAMPLE CASE {i} FAILED =====")
            print(repr(e))
            continue

        choices = data.get("choices") or []
        usage = data.get("usage")
        print(f"\n===== SAMPLE CASE {i} =====")
        print(f"user: {_user_preview(messages)}")
        print(f"choices={len(choices)} usage={usage}")
        show = list(range(min(top_k, len(choices))))
        if len(choices) > top_k:
            show.append(len(choices) - 1)
        for j in show:
            c = choices[j]
            msg = c.get("message") or {}
            content = msg.get("content")
            score = (c.get("sglext") or {}).get("sequence_score")
            label = sid_validity_label(content, valid_sids)
            label_s = f" sid={label}" if label else ""
            print(
                f"  [{j:02d}] finish={c.get('finish_reason')} "
                f"score={score}{label_s} content={content!r}"
            )
        if len(choices) > top_k + 1:
            print(f"  ... ({len(choices) - top_k - 1} more beams omitted)")
    print("\n[sample] done, starting benchmark...\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--url", default="http://127.0.0.1:8904/v1/chat/completions")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--parallel", nargs="+", default=["1"])
    p.add_argument("--number", nargs="+", default=["5000"])
    p.add_argument("--warmup-num", type=float, default=10)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=5,
        help="生成长度（对应 curl 的 max_completion_tokens）",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--n",
        type=int,
        default=50,
        help="beam width（对应 curl 的 n）",
    )
    p.add_argument(
        "--print-samples",
        type=int,
        default=3,
        help="压测前打印多少条真实返回 case，0 关闭",
    )
    p.add_argument(
        "--print-top-k",
        type=int,
        default=5,
        help="每条 sample 打印前 K 个 beam（另附最后一个）",
    )
    p.add_argument("--outputs-dir", type=Path, default=BENCH_DIR / "outputs")
    p.add_argument("--name", default="genrec_v1.1.23_fp8")
    p.add_argument(
        "--shared-system-prompt",
        default=None,
        help=(
            "可选：写入临时 jsonl，给每条样本前置固定 system，用于 radix 前缀 A/B。"
            "也可不设本参数、改用服务端 --beam-shared-system-prompt。"
        ),
    )
    p.add_argument(
        "--shared-system-prompt-file",
        type=Path,
        default=None,
        help="从文件读取 --shared-system-prompt（UTF-8）。",
    )
    p.add_argument(
        "--valid-sid-file",
        type=Path,
        default=DEFAULT_VALID_SID_FILE,
        help=(
            "合法 SID 集合文件（sid2vid.json 或 SID 列表）。"
            "用于计算生成结果是否落在指定 SID 集合中的合法率。"
        ),
    )
    p.add_argument(
        "--no-sid-valid-check",
        action="store_true",
        help="关闭压测前的 SID 合法率检查。",
    )
    p.add_argument(
        "--sid-valid-num",
        type=int,
        default=100,
        help="合法率检查使用的请求数（从 dataset 取前 N 条）；0 关闭。",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if not args.dataset_path.exists():
        raise SystemExit(f"dataset not found: {args.dataset_path}")

    valid_sids: Optional[Set[str]] = None
    if not args.no_sid_valid_check and args.sid_valid_num > 0:
        if args.valid_sid_file is None:
            raise SystemExit("--sid-valid-num>0 时需要提供 --valid-sid-file")
        if not args.valid_sid_file.exists():
            raise SystemExit(f"valid SID file not found: {args.valid_sid_file}")
        print(f"[sid-valid] loading SID set from {args.valid_sid_file} ...")
        valid_sids = load_valid_sid_set(args.valid_sid_file)
        print(f"[sid-valid] loaded {len(valid_sids)} SIDs")

    shared_system = args.shared_system_prompt
    if args.shared_system_prompt_file is not None:
        if not args.shared_system_prompt_file.exists():
            raise SystemExit(
                f"shared system prompt file not found: {args.shared_system_prompt_file}"
            )
        shared_system = args.shared_system_prompt_file.read_text(encoding="utf-8").strip()
        if not shared_system:
            raise SystemExit(
                f"shared system prompt file is empty: {args.shared_system_prompt_file}"
            )

    dataset_path = args.dataset_path
    if shared_system:
        dataset_path = _write_dataset_with_shared_system(
            args.dataset_path, shared_system, args.outputs_dir
        )
        print(
            f"[info] wrote dataset with shared system ({len(shared_system)} chars) "
            f"-> {dataset_path}"
        )

    parallel = parse_int_list(args.parallel)
    number = parse_int_list(args.number)
    if len(number) == 1 and len(parallel) > 1:
        number = number * len(parallel)
    if len(parallel) != len(number):
        raise SystemExit(f"--parallel {parallel} 与 --number {number} 长度需一致")

    print_sample_cases(
        url=args.url,
        model=args.model,
        dataset_path=dataset_path,
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sample_num=args.print_samples,
        top_k=args.print_top_k,
        valid_sids=valid_sids,
    )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    if valid_sids is not None and args.sid_valid_num > 0:
        stats = compute_sid_validity(
            url=args.url,
            model=args.model,
            dataset_path=dataset_path,
            n=args.n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_requests=args.sid_valid_num,
            valid_sids=valid_sids,
        )
        report_path = args.outputs_dir / "sid_validity.json"
        print_sid_validity_report(stats, out_path=report_path)

    task_cfg = Arguments(
        model=args.model,
        url=args.url,
        api="openai",
        api_key=args.api_key,
        dataset="line_by_line",
        dataset_path=str(dataset_path.resolve()),
        tokenizer_path=args.tokenizer_path or args.model,
        parallel=parallel,
        number=number,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=True,
        warmup_num=args.warmup_num,
        outputs_dir=str(args.outputs_dir),
        name=args.name,
        n_choices=args.n,
        extra_args={
            "max_completion_tokens": args.max_tokens,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "separate_reasoning": False,
        },
        debug=args.debug,
    )
    _patch_evalscope_single_turn_cache_reporting()
    print(json.dumps(task_cfg.model_dump(), ensure_ascii=False, indent=2, default=str))
    run_perf_benchmark(task_cfg)
    print(f"[ok] done, outputs -> {args.outputs_dir}")


if __name__ == "__main__":
    main()
