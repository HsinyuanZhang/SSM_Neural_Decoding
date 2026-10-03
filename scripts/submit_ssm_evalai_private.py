#!/usr/bin/env python3
"""Fail-closed, one-shot private EvalAI controller.

The default operation is entirely read-only.  ``--execute`` is the sole path
that pushes an already-built image and creates a submission.  It deliberately
records a durable image URI and a ``post_attempted`` marker before POSTing, so
an interrupted request can never be retried blindly.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

API = "https://eval.ai"
CHALLENGE = 2319
PHASE = 4599
TEAM = 42279
TEAM_NAME = "sustechhku"
USERNAME = "Federico_Wen"
ECR_REPOSITORY = "few-shot-algorithms-for-consistent-neural-decoding-falcon-2319-participant-team-42279"
ACTIVE = frozenset({"submitted", "submitting", "resuming", "queued", "running"})
SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
REFERENCE_TOKEN = Path.home() / ".evalai/token.json.sustechhku_20260911"
CURRENT_TOKEN = Path.home() / ".evalai/token.json"
ROOT = Path(__file__).resolve().parents[1]
GLOBAL_LOCK = ROOT / "results/official_evalai/.team42279_phase4599.lock"


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def save(path: Path | str, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def bearer() -> str:
    """Compare each token file once, entirely in memory."""
    reference = load_object(REFERENCE_TOKEN).get("token")
    current = load_object(CURRENT_TOKEN).get("token")
    if not isinstance(reference, str) or not reference or current != reference:
        raise RuntimeError("current EvalAI token does not match sustechhku reference")
    return reference


def get_json(path: str, token: str) -> Any:
    response = requests.get(API + path, headers={"Authorization": f"Bearer {token}"}, timeout=(20, 60))
    response.raise_for_status()
    return response.json()


def list_submissions(token: str) -> list[dict[str, Any]]:
    path: str | None = f"/api/jobs/challenge/{CHALLENGE}/challenge_phase/{PHASE}/submission/"
    result: list[dict[str, Any]] = []
    while path:
        page = get_json(path, token)
        if not isinstance(page, dict):
            raise RuntimeError("invalid submission-list response")
        result.extend(row for row in page.get("results", []) if isinstance(row, dict))
        next_url = page.get("next")
        parsed = urlparse(str(next_url)) if next_url else None
        path = parsed.path + (("?" + parsed.query) if parsed and parsed.query else "") if parsed else None
    return result


def verify_digest(binding: dict[str, Any], path_key: str, label: str) -> Path:
    path = binding.get(path_key)
    expected = binding.get("sha256")
    if not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{label} hash binding malformed")
    candidate = Path(path)
    if not candidate.is_file() or sha256(candidate) != expected:
        raise RuntimeError(f"{label} hash mismatch")
    return candidate


def read_pass_report(path: Path, section: dict[str, Any], name: str) -> dict[str, Any]:
    if section.get("status") != "PASS":
        raise RuntimeError(f"{name} manifest status is not PASS")
    report = load_object(path)
    if report.get("status") != "PASS":
        raise RuntimeError(f"{name} report status is not PASS")
    return report


def audit_manifest(path: Path | str, image_tag: str, task: str) -> dict[str, Any]:
    """Validate every local artifact named by the candidate audit manifest."""
    manifest = load_object(path)
    required = {"candidate_source", "payload", "image", "cpu_gpu_parity", "sdk_container_preflight"}
    if manifest.get("schema") != "ssm_evalai_candidate_audit_v1" or manifest.get("task") != task:
        raise RuntimeError("candidate audit schema/task mismatch")
    if not isinstance(manifest.get("method_name"), str) or not manifest["method_name"]:
        raise RuntimeError("candidate audit method name missing")
    if not required.issubset(manifest):
        raise RuntimeError("candidate audit sections missing")
    for section, key, label in (
        ("candidate_source", "best_checkpoint", "checkpoint"),
        ("payload", "manifest", "payload manifest"),
        ("cpu_gpu_parity", "report", "CPU/GPU parity report"),
        ("sdk_container_preflight", "report", "SDK/container preflight report"),
    ):
        if not isinstance(manifest[section], dict):
            raise RuntimeError(f"{label} binding missing")
        verify_digest(manifest[section], key, label)
    image = manifest["image"]
    if not isinstance(image, dict) or image.get("tag") != image_tag:
        raise RuntimeError("command image tag differs from audit manifest")
    image_id = image.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:") or len(image_id) != 71:
        raise RuntimeError("audit image ID malformed")
    parity_path = Path(manifest["cpu_gpu_parity"]["report"])
    parity = read_pass_report(parity_path, manifest["cpu_gpu_parity"], "CPU/GPU parity")
    delta = parity.get("score_delta")
    if parity.get("scope") != "full_query" or isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise RuntimeError("CPU/GPU parity does not cover full query")
    if not math.isfinite(delta) or abs(delta) > 1e-3:
        raise RuntimeError("CPU/GPU score delta exceeds 1e-3")
    preflight = read_pass_report(
        Path(manifest["sdk_container_preflight"]["report"]),
        manifest["sdk_container_preflight"],
        "SDK/container preflight",
    )
    if preflight.get("scope") not in {"official_sdk_container", "official_sdk_container_full_query"}:
        raise RuntimeError("SDK/container preflight scope missing")
    return manifest


def local_image_id(image_tag: str) -> str:
    import docker

    value = docker.from_env().images.get(image_tag).id
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise RuntimeError("Docker returned malformed image ID")
    return value


def validate_local_image(manifest: dict[str, Any], image_tag: str) -> None:
    if local_image_id(image_tag) != manifest["image"]["image_id"]:
        raise RuntimeError("local Docker image ID differs from audit manifest")


def identity_and_quota(token: str) -> dict[str, Any]:
    user = get_json("/api/auth/user/", token)
    if not isinstance(user, dict) or user.get("username") != USERNAME:
        raise RuntimeError("EvalAI account identity mismatch")
    document = get_json(f"/api/jobs/{CHALLENGE}/remaining_submissions/", token)
    if not isinstance(document, dict):
        raise RuntimeError("EvalAI quota response is not an object")
    if document.get("participant_team_id") != TEAM or document.get("participant_team") != TEAM_NAME:
        raise RuntimeError("EvalAI authoritative quota document has wrong team")
    phases = document.get("phases", []) if isinstance(document, dict) else []
    matches = [row for row in phases if isinstance(row, dict) and row.get("id") == PHASE]
    if len(matches) != 1 or not isinstance(matches[0].get("limits"), dict):
        raise RuntimeError("EvalAI quota response has no target phase")
    limits = matches[0]["limits"]
    required = (
        "remaining_submissions_this_month_count",
        "remaining_submissions_today_count",
        "remaining_submissions_count",
    )
    quota = {key: limits.get(key) for key in required}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in quota.values()):
        raise RuntimeError("EvalAI quota counts unavailable or exhausted")
    return quota


def ours(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("participant_team") == TEAM and row.get("challenge_phase") == PHASE]


def safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "id", "participant_team", "participant_team_name", "challenge_phase", "status", "is_public",
        "method_name", "submitted_at", "started_at", "completed_at", "execution_time",
    )}


def server_markers(row: dict[str, Any]) -> tuple[str, str] | None:
    description = row.get("method_description")
    if not isinstance(description, str):
        return None
    match = re.fullmatch(r"SSM private candidate image_uri=([^;\s]+); audit_manifest_sha256=([0-9a-f]{64})", description)
    return (match.group(1), match.group(2)) if match else None


def candidate_rows(team_rows: list[dict[str, Any]], audit_sha: str) -> list[dict[str, Any]]:
    matches = []
    for row in team_rows:
        markers = server_markers(row)
        if markers is not None and markers[1] == audit_sha:
            matches.append(row)
    return matches


def confirmed_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def allow_authenticated_active_intents(
    paths: list[Path], active_rows: list[dict[str, Any]], current_method_name: str, current_audit_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return only active rows authenticated by explicit, confirmed prior intents."""
    if not isinstance(current_method_name, str) or not current_method_name:
        raise RuntimeError("current candidate method name missing")
    if not isinstance(current_audit_sha, str) or re.fullmatch(r"[0-9a-f]{64}", current_audit_sha) is None:
        raise RuntimeError("current candidate audit hash malformed")
    allowed: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_intent_hashes: set[str] = set()
    seen_submission_ids: set[int] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen_paths:
            raise RuntimeError("allow-active intent path is repeated")
        seen_paths.add(path)
        if not path.is_file():
            raise RuntimeError("allow-active intent path does not exist")
        initial_sha = sha256(path)
        if initial_sha in seen_intent_hashes:
            raise RuntimeError("allow-active intent content is repeated")
        seen_intent_hashes.add(initial_sha)
        state = load_object(path)
        if sha256(path) != initial_sha:
            raise RuntimeError("allow-active intent changed while being read")
        if (
            state.get("schema") != "ssm_evalai_private_intent_v1"
            or isinstance(state.get("team"), bool)
            or not isinstance(state.get("team"), int)
            or state.get("team") != TEAM
            or isinstance(state.get("phase"), bool)
            or not isinstance(state.get("phase"), int)
            or state.get("phase") != PHASE
            or state.get("post_attempted") is not True
            or not confirmed_utc(state.get("post_confirmed_utc"))
        ):
            raise RuntimeError("allow-active intent is not a confirmed private submission intent")
        submission_id = state.get("submission_id")
        uri, audit_sha, method_name = (
            state.get("submitted_image_uri"), state.get("candidate_audit_sha256"), state.get("method_name")
        )
        if (
            isinstance(submission_id, bool)
            or not isinstance(submission_id, int)
            or submission_id < 1
            or not isinstance(uri, str)
            or not uri
            or not isinstance(audit_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", audit_sha) is None
            or not isinstance(method_name, str)
            or not method_name
        ):
            raise RuntimeError("allow-active intent lacks exact server reconciliation bindings")
        if method_name == current_method_name or audit_sha == current_audit_sha:
            raise RuntimeError("allow-active intent must be a different method and candidate audit")
        if submission_id in seen_submission_ids:
            raise RuntimeError("allow-active intents authenticate the same submission ID")
        seen_submission_ids.add(submission_id)
        matches = [row for row in active_rows if row.get("id") == submission_id]
        if len(matches) != 1:
            raise RuntimeError("allow-active intent does not match exactly one active server row")
        row = matches[0]
        if (
            isinstance(row.get("id"), bool)
            or not isinstance(row.get("id"), int)
            or row.get("id") != submission_id
            or row.get("status") not in ACTIVE
            or row.get("is_public") is not False
            or isinstance(row.get("participant_team"), bool)
            or not isinstance(row.get("participant_team"), int)
            or row.get("participant_team") != TEAM
            or isinstance(row.get("challenge_phase"), bool)
            or not isinstance(row.get("challenge_phase"), int)
            or row.get("challenge_phase") != PHASE
            or row.get("method_name") != method_name
            or server_markers(row) != (uri, audit_sha)
        ):
            raise RuntimeError("allow-active intent and active server row differ")
        allowed.append(row)
        receipts.append({"intent_path": str(path), "intent_sha256": initial_sha, "submission_id": submission_id})
    return allowed, receipts


def reconcile_prior_intent(intent: Path, team_rows: list[dict[str, Any]], receipts: Path) -> bool:
    """Return true only for a server-confirmed prior post; reject uncertainty."""
    if not intent.exists():
        return False
    state = load_object(intent)
    uri, audit_sha, method_name = state.get("submitted_image_uri"), state.get("candidate_audit_sha256"), state.get("method_name")
    if not all(isinstance(value, str) and value for value in (uri, audit_sha, method_name)):
        raise RuntimeError("prior intent lacks exact reconciliation bindings")
    matches = [row for row in team_rows if row.get("method_name") == method_name and server_markers(row) == (uri, audit_sha)]
    if len(matches) == 1:
        row = matches[0]
        if row.get("participant_team") != TEAM or row.get("challenge_phase") != PHASE or row.get("is_public") is not False:
            raise RuntimeError("prior URI reconciled to wrong/private-invalid submission")
        save(receipts / "reconciled.json", {"reconciled_utc": utcnow(), "submission": safe_row(row)})
        return True
    if len(matches) > 1:
        raise RuntimeError("prior URI reconciliation is ambiguous")
    if state.get("post_attempted"):
        raise RuntimeError("uncertain previous POST: refuse a second POST until exact URI reconciliation")
    raise RuntimeError("previous upload intent exists: refuse duplicate upload/post")


def parse_ecr_binding(document: Any) -> dict[str, Any]:
    """Validate a read-only EvalAI response without retaining printable secrets."""
    value = document.get("success") if isinstance(document, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("EvalAI ECR credential request failed")
    repository = value.get("docker_repository_uri")
    if not isinstance(repository, str) or repository.rsplit("/", 1)[-1] != ECR_REPOSITORY:
        raise RuntimeError("EvalAI ECR repository does not belong to target team")
    federated = value.get("federated_user")
    credentials = federated.get("Credentials") if isinstance(federated, dict) else None
    if not isinstance(credentials, dict) or not all(isinstance(credentials.get(k), str) and credentials[k] for k in ("AccessKeyId", "SecretAccessKey", "SessionToken")):
        raise RuntimeError("EvalAI ECR credentials malformed")
    federated_id = federated.get("FederatedUser", {}).get("FederatedUserId", "") if isinstance(federated, dict) else ""
    account = str(federated_id).split(":", 1)[0]
    if not account:
        raise RuntimeError("EvalAI ECR account binding malformed")
    registry = repository.split("/", 1)[0]
    expected_repository = f"{account}.dkr.ecr.us-east-1.amazonaws.com/{ECR_REPOSITORY}"
    if repository != expected_repository:
        raise RuntimeError("EvalAI ECR registry account/region binding mismatch")
    return {"repository": repository, "account": account, "credentials": credentials}


def ecr_binding(token: str) -> dict[str, Any]:
    """Fetch the temporary registry binding and create an in-memory ECR client."""
    parsed = parse_ecr_binding(get_json(f"/api/challenges/phases/{PHASE}/participant_team/aws/credentials/", token))
    import boto3

    client = boto3.client(
        "ecr", region_name="us-east-1", aws_access_key_id=parsed["credentials"]["AccessKeyId"],
        aws_secret_access_key=parsed["credentials"]["SecretAccessKey"], aws_session_token=parsed["credentials"]["SessionToken"],
    )
    return {"repository": parsed["repository"], "account": parsed["account"], "client": client}


def required_sha256_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_DIGEST.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a canonical SHA-256 digest")
    return value


def pushed_manifest_digest(events: Any, tag: str) -> str:
    """Return Docker 29's one canonical final tag/digest status.

    Docker 29 reports the pushed manifest in ``status`` as
    ``<tag>: digest: sha256:... size: ...``.  Older engines can additionally
    emit an ``aux.Digest`` event; it is only corroborating metadata and must
    agree exactly when present.
    """
    if not isinstance(tag, str) or not tag:
        raise RuntimeError("Docker push tag missing")
    final = re.compile(
        rf"{re.escape(tag)}: digest: (sha256:[0-9a-f]{{64}}) size: (?:0|[1-9][0-9]*)"
    )
    status_digests: list[str] = []
    aux_digests: list[str] = []
    try:
        iterator = iter(events)
    except TypeError as error:
        raise RuntimeError("Docker push stream is not iterable") from error
    for event in iterator:
        if not isinstance(event, dict):
            raise RuntimeError("Docker push stream event is malformed")
        if event.get("error") is not None or event.get("errorDetail") is not None:
            raise RuntimeError("Docker image push failed")
        if "status" in event:
            status = event["status"]
            if not isinstance(status, str):
                raise RuntimeError("Docker push status is malformed")
            if "digest:" in status:
                match = final.fullmatch(status)
                if match is None:
                    raise RuntimeError("Docker push final tag/digest status is malformed")
                status_digests.append(match.group(1))
        if "aux" in event:
            aux = event["aux"]
            if not isinstance(aux, dict):
                raise RuntimeError("Docker push auxiliary status is malformed")
            aux_digest = required_sha256_digest(aux.get("Digest"), "Docker push auxiliary digest")
            aux_tag = aux.get("Tag")
            if aux_tag is not None and aux_tag != tag:
                raise RuntimeError("Docker push auxiliary tag differs from pushed tag")
            aux_digests.append(aux_digest)
    if len(status_digests) != 1:
        raise RuntimeError("Docker push must contain exactly one final tag/digest status")
    digest = status_digests[0]
    if any(aux_digest != digest for aux_digest in aux_digests):
        raise RuntimeError("Docker push status and auxiliary digest disagree")
    return digest


def ecr_pushed_manifest(ecr: Any, repository_name: str, tag: str, manifest_digest: str) -> dict[str, Any]:
    """Read one ECR manifest and bind its bytes, tag, and digest exactly."""
    response = ecr.batch_get_image(
        repositoryName=repository_name,
        imageIds=[{"imageTag": tag}],
        acceptedMediaTypes=[
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
        ],
    )
    if not isinstance(response, dict) or response.get("failures"):
        raise RuntimeError("ECR did not return the pushed manifest")
    images = response.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError("ECR did not return exactly one pushed manifest")
    image = images[0]
    image_id = image.get("imageId")
    if not isinstance(image_id, dict) or image_id.get("imageTag") != tag:
        raise RuntimeError("ECR manifest tag differs from pushed tag")
    if required_sha256_digest(image_id.get("imageDigest"), "ECR manifest image digest") != manifest_digest:
        raise RuntimeError("ECR manifest digest differs from pushed image digest")
    encoded = image.get("imageManifest")
    if not isinstance(encoded, str):
        raise RuntimeError("ECR manifest body missing")
    try:
        manifest = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise RuntimeError("ECR manifest body is malformed") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2:
        raise RuntimeError("ECR manifest schema is not version 2")
    body_media_type = manifest.get("mediaType")
    if not isinstance(body_media_type, str) or not body_media_type:
        raise RuntimeError("ECR manifest body media type missing")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("ECR manifest config missing")
    media_type = image.get("imageManifestMediaType")
    if not isinstance(media_type, str) or not media_type:
        raise RuntimeError("ECR manifest media type missing")
    if body_media_type != media_type:
        raise RuntimeError("ECR manifest body and response media type disagree")
    manifest_bytes = encoded.encode("utf-8")
    actual_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    if actual_digest != manifest_digest:
        raise RuntimeError("ECR manifest bytes digest differs from pushed image digest")
    return {
        "digest": actual_digest,
        "config_digest": required_sha256_digest(config.get("digest"), "ECR manifest config digest"),
        "media_type": media_type,
        "size": len(manifest_bytes),
    }


def validate_pushed_manifest_identity(
    expected_image_id: str, local_descriptor: Any, pushed_manifest: dict[str, Any]
) -> None:
    """Bind the local Docker 29 descriptor, or classic config, to ECR bytes."""
    expected_image_id = required_sha256_digest(expected_image_id, "audited local image ID")
    if not isinstance(pushed_manifest, dict):
        raise RuntimeError("pushed ECR manifest binding malformed")
    manifest_digest = required_sha256_digest(pushed_manifest.get("digest"), "ECR manifest digest")
    if local_descriptor is None:
        if pushed_manifest.get("config_digest") != expected_image_id:
            raise RuntimeError("ECR pushed manifest config digest differs from audited local image ID")
        return
    if not isinstance(local_descriptor, dict):
        raise RuntimeError("local Docker image Descriptor is malformed")
    descriptor_digest = required_sha256_digest(local_descriptor.get("digest"), "local Docker image Descriptor digest")
    descriptor_media_type = local_descriptor.get("mediaType")
    descriptor_size = local_descriptor.get("size")
    if not isinstance(descriptor_media_type, str) or not descriptor_media_type:
        raise RuntimeError("local Docker image Descriptor media type is malformed")
    if isinstance(descriptor_size, bool) or not isinstance(descriptor_size, int) or descriptor_size < 1:
        raise RuntimeError("local Docker image Descriptor size is malformed")
    if descriptor_digest != expected_image_id or manifest_digest != expected_image_id:
        raise RuntimeError("local Docker Descriptor, ECR manifest, and audited image ID disagree")
    if descriptor_media_type != pushed_manifest.get("media_type"):
        raise RuntimeError("local Docker Descriptor and ECR manifest media type disagree")
    if descriptor_size != pushed_manifest.get("size"):
        raise RuntimeError("local Docker Descriptor and ECR manifest size disagree")


def push_image(image_tag: str, expected_image_id: str, uri: str, binding: dict[str, Any]) -> None:
    """Tag and push; output is intentionally suppressed to avoid credential/URI leakage."""
    import docker

    client = docker.from_env()
    image = client.images.get(image_tag)
    if image.id != expected_image_id:
        raise RuntimeError("local image changed after audit")
    image_attrs = getattr(image, "attrs", None)
    if image_attrs is not None and not isinstance(image_attrs, dict):
        raise RuntimeError("local Docker image inspection is malformed")
    local_descriptor = image_attrs.get("Descriptor") if isinstance(image_attrs, dict) else None
    repo, tag = uri.rsplit(":", 1)
    ecr = binding["client"]
    auth = ecr.get_authorization_token(registryIds=[binding["account"]])["authorizationData"][0]
    username, password = base64.b64decode(auth["authorizationToken"]).decode("utf-8").split(":", 1)
    client.login(username=username, password=password, registry=auth["proxyEndpoint"], reauth=True)
    image.tag(repo, tag=tag)
    pushed_digest = pushed_manifest_digest(
        client.images.push(repo, tag=tag, stream=True, decode=True), tag
    )
    repository_name = repo.split("/", 1)[1]
    details = ecr.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])
    image_details = details.get("imageDetails") if isinstance(details, dict) else None
    if not isinstance(image_details, list) or len(image_details) != 1:
        raise RuntimeError("ECR did not confirm uploaded image")
    ecr_digest = image_details[0].get("imageDigest") if isinstance(image_details[0], dict) else None
    if required_sha256_digest(ecr_digest, "ECR image digest") != pushed_digest:
        raise RuntimeError("Docker push and ECR image digest disagree")
    pushed_manifest = ecr_pushed_manifest(ecr, repository_name, tag, pushed_digest)
    validate_pushed_manifest_identity(expected_image_id, local_descriptor, pushed_manifest)


def post_private(token: str, manifest: dict[str, Any], uri: str, receipt_dir: Path) -> dict[str, Any]:
    description = f"SSM private candidate image_uri={uri}; audit_manifest_sha256={manifest['_audit_sha256']}"
    payload = {
        "method_name": manifest["method_name"],
        "method_description": description,
        "is_public": json.dumps(False),
        "status": "submitting",
    }
    upload = receipt_dir / ".submission_image_uri.json"
    save(upload, {"submitted_image_uri": uri})
    with upload.open("rb") as handle:
        response = requests.post(
            API + f"/api/jobs/challenge/{CHALLENGE}/challenge_phase/{PHASE}/submission/",
            headers={"Authorization": f"Bearer {token}"}, files={"input_file": handle}, data=payload, timeout=(20, 120),
        )
    response.raise_for_status()
    answer = response.json()
    if not isinstance(answer, dict):
        raise RuntimeError("invalid EvalAI POST response")
    return answer


def verify_post(answer: dict[str, Any]) -> dict[str, Any]:
    if answer.get("participant_team") != TEAM or answer.get("challenge_phase") != PHASE:
        raise RuntimeError("EvalAI POST returned wrong team or phase")
    if answer.get("is_public") is not False:
        raise RuntimeError("EvalAI POST did not create a private submission")
    if not isinstance(answer.get("id"), int):
        raise RuntimeError("EvalAI POST response lacks submission ID")
    return safe_row(answer)


def get_submission_detail(token: str, submission_id: int) -> dict[str, Any]:
    detail = get_json(f"/api/jobs/submission/{submission_id}", token)
    if not isinstance(detail, dict):
        raise RuntimeError("submission detail response is not an object")
    if detail.get("id") != submission_id or detail.get("participant_team") != TEAM or detail.get("challenge_phase") != PHASE:
        raise RuntimeError("submission detail changed team or phase")
    if detail.get("is_public") is not False:
        raise RuntimeError("submission detail is not private")
    return detail


def safe_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Persist finite score primitives and approved partition descriptors only."""
    sensitive = ("url", "uri", "file", "token", "secret", "credential", "password", "authorization")
    descriptor_keys = frozenset({"split", "partition", "dataset", "task", "session", "session_label"})
    descriptor_value = re.compile(r"^(?:test_split_[a-z0-9_]+|held_(?:in|out)|(?:train|test|validation)(?:_[a-z0-9_]+)?|[mh][0-9]+|session_[a-z0-9_]+)$")
    score_key = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

    def redact(value: Any, key: str = "") -> Any:
        if any(word in key.lower() for word in sensitive):
            return None
        if isinstance(value, str):
            return value if key in descriptor_keys and descriptor_value.fullmatch(value) else None
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value if math.isfinite(value) else None
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {
                str(name): cleaned for name, item in value.items()
                if score_key.fullmatch(str(name)) and (cleaned := redact(item, str(name))) is not None
            }
        return None

    return {**safe_row(detail), "result": redact(detail.get("result"))}


def poll_until_terminal(token: str, submission_id: int, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        detail = get_submission_detail(token, submission_id)
        if detail.get("status") not in ACTIVE:
            return safe_detail(detail)
        if time.monotonic() >= deadline:
            raise RuntimeError("poll timeout; receipt preserves submission ID for --poll-existing")
        print(json.dumps({"stage": "POLL", "submission_id": submission_id, "status": detail.get("status"), "is_public": False}, sort_keys=True), flush=True)
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--task", choices=("m1", "m2"))
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--receipt-dir", type=Path, default=Path("results/official_evalai"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--poll-existing", type=int)
    parser.add_argument("--allow-active-intent", type=Path, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    token = bearer()
    if args.poll_existing is not None:
        detail = get_submission_detail(token, args.poll_existing)
        save(args.receipt_dir / f"poll_{args.poll_existing}.json", {"polled_utc": utcnow(), "submission": safe_detail(detail)})
        print(json.dumps({"stage": "POLL", "submission_id": args.poll_existing, "status": detail.get("status"), "is_public": False}, sort_keys=True), flush=True)
        return
    if not (args.image and args.task and args.candidate_manifest):
        raise SystemExit("--image, --task, and --candidate-manifest are required")
    if args.poll_timeout_seconds < 1:
        raise SystemExit("--poll-timeout-seconds must be positive")
    GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOCK.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = audit_manifest(args.candidate_manifest, args.image, args.task)
        manifest["_audit_sha256"] = sha256(args.candidate_manifest)
        validate_local_image(manifest, args.image)
        quota = identity_and_quota(token)
        team_rows = ours(list_submissions(token))
        intent = args.receipt_dir / "intent.json"
        active = [row for row in team_rows if row.get("status") in ACTIVE]
        allowed_active, allowed_active_intents = allow_authenticated_active_intents(
            args.allow_active_intent, active, manifest["method_name"], manifest["_audit_sha256"]
        )
        allowed_active_ids = {row["id"] for row in allowed_active}
        blocked_active = [row for row in active if row.get("id") not in allowed_active_ids]
        duplicate_audit = candidate_rows(team_rows, manifest["_audit_sha256"])
        duplicate_method = [row for row in team_rows if row.get("method_name") == manifest["method_name"]]
        save(args.receipt_dir / "preflight.json", {
            "preflight_utc": utcnow(), "team": TEAM, "phase": PHASE, "quota": quota,
            "active_submission_ids": [row.get("id") for row in active],
            "allowed_active_submission_ids": [row.get("id") for row in allowed_active],
            "blocked_active_submission_ids": [row.get("id") for row in blocked_active],
            "allowed_active_intents": allowed_active_intents,
            "candidate_audit_duplicate_submission_ids": [row.get("id") for row in duplicate_audit],
            "method_name_duplicate_submission_ids": [row.get("id") for row in duplicate_method],
            "audit_manifest_sha256": manifest["_audit_sha256"], "local_image_id": manifest["image"]["image_id"],
        })
        if reconcile_prior_intent(intent, team_rows, args.receipt_dir):
            return
        if args.preflight or not args.execute:
            return
        if blocked_active or duplicate_audit or duplicate_method:
            raise RuntimeError("active or duplicate team/phase submission prevents POST")
        binding = ecr_binding(token)
        image_uri = f"{binding['repository']}:{uuid.uuid4().hex}"
        save(intent, {
            "schema": "ssm_evalai_private_intent_v1", "created_utc": utcnow(), "team": TEAM, "phase": PHASE,
            "image_tag": args.image, "image_id": manifest["image"]["image_id"], "submitted_image_uri": image_uri,
            "candidate_audit_sha256": manifest["_audit_sha256"], "method_name": manifest["method_name"], "post_attempted": False,
        })
        push_image(args.image, manifest["image"]["image_id"], image_uri, binding)
        state = load_object(intent); state["uploaded_utc"] = utcnow(); save(intent, state)
        state["post_attempted"] = True; state["post_attempted_utc"] = utcnow(); save(intent, state)
        answer = post_private(token, manifest, image_uri, args.receipt_dir)
        receipt = verify_post(answer)
        state["submission_id"] = receipt["id"]; state["post_confirmed_utc"] = utcnow(); save(intent, state)
        save(args.receipt_dir / f"submission_{receipt['id']}.json", {"receipt_utc": utcnow(), "submission": receipt})
        print(json.dumps({"stage": "POSTED", "submission_id": receipt["id"], "status": receipt.get("status"), "is_public": False, "team": TEAM, "phase": PHASE}, sort_keys=True), flush=True)
        terminal = poll_until_terminal(token, receipt["id"], args.poll_timeout_seconds)
        save(args.receipt_dir / f"terminal_{receipt['id']}.json", {"terminal_utc": utcnow(), "submission": terminal})
        print(json.dumps({"stage": "TERMINAL", "submission_id": receipt["id"], "status": terminal.get("status"), "is_public": False, "team": TEAM, "phase": PHASE}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
