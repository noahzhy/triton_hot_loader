"""Filesystem half of version removal; also runs in the maintenance image.

Only Python's standard library is required. Plans are persisted by the controller
before prepare. Every modifying action is restartable using that same plan.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

PROTOCOL = 1
RESULT_PREFIX = "VERSION_TRANSACTION_RESULT="


def atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, name = tempfile.mkstemp(prefix=".version-config-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def version_policy(config: str, versions: list[str]) -> str:
    # Tokenize comments and strings as units so braces inside them do not count.
    tokens = list(re.finditer(r'#[^\n]*|//[^\n]*|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|[A-Za-z_][\w]*|[^\s]', config))
    ranges = []
    depth = 0
    i = 0
    while i < len(tokens):
        token = tokens[i].group()
        if token == "version_policy" and depth == 0:
            start = tokens[i].start()
            i += 1
            if i < len(tokens) and tokens[i].group() == ":":
                i += 1
            if i >= len(tokens) or tokens[i].group() not in {"{", "<"}:
                raise ValueError("无法解析 version_policy")
            block_depth = 1
            i += 1
            while i < len(tokens) and block_depth:
                value = tokens[i].group()
                block_depth += int(value in {"{", "<"}) - int(value in {"}", ">"})
                i += 1
            if block_depth:
                raise ValueError("version_policy 未闭合")
            ranges.append((start, tokens[i - 1].end()))
            continue
        depth += int(token in {"{", "<"}) - int(token in {"}", ">"})
        i += 1
    for start, end in reversed(ranges):
        config = config[:start] + config[end:]
    return config.rstrip() + "\nversion_policy: { specific { versions: [ " + ", ".join(versions) + " ] } }\n"


def _no_links(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"拒绝符号链接: {path}")
    if path.is_dir():
        for current, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                child = Path(current) / name
                if child.is_symlink():
                    raise ValueError(f"拒绝符号链接: {child}")


def locations(root: str, model: str, operation_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-z0-9_-]+", model) or not re.fullmatch(r"[a-f0-9]{32}", operation_id):
        raise ValueError("非法模型名或 operation_id")
    repository = Path(root).absolute()
    if repository.parent == repository or repository.is_symlink() or not repository.is_dir():
        raise ValueError(f"仓库不存在或为符号链接: {repository}")
    model_dir = repository / model
    _no_links(model_dir)
    if not model_dir.is_dir():
        raise ValueError(f"模型目录不存在: {model_dir}")
    backup_base = repository.parent / ".hot-loader-version-backups"
    backup = backup_base / operation_id / model
    for candidate in (backup_base, backup.parent, backup):
        if candidate.is_symlink():
            raise ValueError(f"拒绝符号链接备份路径: {candidate}")
    _no_links(backup)
    ancestor = backup
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != model_dir.stat().st_dev:
        raise ValueError("备份目录必须与模型位于同一文件系统；请将仓库放在挂载卷子目录")
    return model_dir, backup


def inspect(root: str, model: str, operation_id: str, versions: list[str]) -> dict:
    model_dir, backup = locations(root, model, operation_id)
    if not versions or any(not re.fullmatch(r"0|[1-9][0-9]*", v) for v in versions):
        raise ValueError("版本必须是规范的非负数字目录名")
    available = sorted((p.name for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=int)
    if not set(versions) <= set(available):
        raise ValueError("指定版本目录不存在")
    remaining = [v for v in available if v not in versions]
    if not remaining:
        raise ValueError("拒绝删除最后一个版本；请使用整模型卸载")
    original = (model_dir / "config.pbtxt").read_text(encoding="utf-8")
    return {"root": str(Path(root).absolute()), "model": model, "operation_id": operation_id,
            "versions": sorted(set(versions), key=int), "remaining": remaining,
            "original_config": original, "new_config": version_policy(original, remaining),
            "backup": str(backup)}


def execute(action: str, plan: dict) -> dict:
    model_dir, backup = locations(plan["root"], plan["model"], plan["operation_id"])
    if str(backup) != plan["backup"]:
        raise ValueError("备份位置与事务记录不一致")
    versions = plan["versions"]
    if any(not re.fullmatch(r"0|[1-9][0-9]*", v) for v in versions):
        raise ValueError("非法版本")
    config = model_dir / "config.pbtxt"
    if config.read_text(encoding="utf-8") not in (plan["original_config"], plan["new_config"]):
        raise ValueError("模型配置被其他进程修改，保留备份等待确认")
    if action == "prepare":
        backup.mkdir(parents=True, exist_ok=True)
        atomic_write(backup / "original-config.pbtxt", plan["original_config"])
        for version in versions:
            source, saved = model_dir / version, backup / version
            if source.exists() and not saved.exists():
                source.rename(saved)
            elif not saved.is_dir() or source.exists():
                raise ValueError(f"版本备份状态不一致: {version}")
        atomic_write(config, plan["new_config"])
    elif action == "restore":
        for version in versions:
            source, saved = model_dir / version, backup / version
            if saved.exists() and not source.exists():
                saved.rename(source)
            elif saved.exists() or not source.is_dir():
                raise ValueError(f"版本恢复状态不一致: {version}")
        atomic_write(config, plan["original_config"])
        if backup.exists():
            shutil.rmtree(backup)
    elif action == "commit":
        if any((model_dir / version).exists() for version in versions):
            raise ValueError("已下线版本重新出现在仓库，拒绝清理备份")
        if config.read_text(encoding="utf-8") != plan["new_config"]:
            raise ValueError("版本策略不一致，拒绝清理备份")
        if backup.exists():
            shutil.rmtree(backup)
    else:
        raise ValueError(f"未知操作: {action}")
    return {"completed": action}


def main() -> None:
    try:
        request = json.loads(os.environ["VERSION_TRANSACTION_REQUEST"])
        if request.get("protocol") != PROTOCOL:
            raise ValueError("维护镜像事务协议不兼容")
        if request["action"] == "inspect":
            result = inspect(**request["arguments"])
        else:
            result = execute(request["action"], request["plan"])
        print(RESULT_PREFIX + json.dumps({"protocol": PROTOCOL, "success": True, "result": result}), flush=True)
    except Exception as exc:
        print(RESULT_PREFIX + json.dumps({"protocol": PROTOCOL, "success": False, "error": str(exc)}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
