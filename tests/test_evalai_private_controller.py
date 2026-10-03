import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "scripts" / "submit_ssm_evalai_private.py"
SPEC = importlib.util.spec_from_file_location("evalai_controller", PATH)
ctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctl)


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")
    return ctl.sha256(path)


def make_manifest(tmp_path, *, delta=0.0, image="repo:test"):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = tmp_path / "payload_manifest.json"
    payload.write_text("payload", encoding="utf-8")
    parity = tmp_path / "parity.json"
    preflight = tmp_path / "preflight.json"
    parity_hash = write_json(parity, {"status": "PASS", "scope": "full_query", "score_delta": delta})
    preflight_hash = write_json(preflight, {"status": "PASS", "scope": "official_sdk_container"})
    manifest = {
        "schema": "ssm_evalai_candidate_audit_v1",
        "task": "m1",
        "method_name": "SSM M1 audited candidate",
        "candidate_source": {"best_checkpoint": str(checkpoint), "sha256": ctl.sha256(checkpoint)},
        "payload": {"manifest": str(payload), "sha256": ctl.sha256(payload)},
        "image": {"tag": image, "image_id": "sha256:" + "a" * 64},
        "cpu_gpu_parity": {"report": str(parity), "sha256": parity_hash, "status": "PASS"},
        "sdk_container_preflight": {"report": str(preflight), "sha256": preflight_hash, "status": "PASS"},
    }
    path = tmp_path / "audit.json"
    write_json(path, manifest)
    return path, manifest


def test_audit_recomputes_hash_rejects_nan_and_tag_drift(tmp_path):
    path, manifest = make_manifest(tmp_path, delta=float("nan"))
    with pytest.raises(RuntimeError, match="score delta"):
        ctl.audit_manifest(path, "repo:test", "m1")

    path, manifest = make_manifest(tmp_path, delta=0.0)
    manifest["payload"]["sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(RuntimeError, match="payload manifest hash mismatch"):
        ctl.audit_manifest(path, "repo:test", "m1")

    path, _ = make_manifest(tmp_path, delta=0.0)
    with pytest.raises(RuntimeError, match="image tag"):
        ctl.audit_manifest(path, "repo:other", "m1")


def test_preflight_report_and_local_image_binding_rejected(tmp_path, monkeypatch):
    path, manifest = make_manifest(tmp_path)
    report = Path(manifest["sdk_container_preflight"]["report"])
    report.write_text(json.dumps({"status": "PASS", "scope": "fixture_only"}), encoding="utf-8")
    manifest["sdk_container_preflight"]["sha256"] = ctl.sha256(report)
    write_json(path, manifest)
    with pytest.raises(RuntimeError, match="scope"):
        ctl.audit_manifest(path, "repo:test", "m1")

    path, manifest = make_manifest(tmp_path)
    checked = ctl.audit_manifest(path, "repo:test", "m1")
    monkeypatch.setattr(ctl, "local_image_id", lambda tag: "sha256:" + "b" * 64)
    with pytest.raises(RuntimeError, match="local Docker image ID"):
        ctl.validate_local_image(checked, "repo:test")


def test_identity_and_quota_are_checked(monkeypatch):
    replies = {
        "/api/auth/user/": {"username": ctl.USERNAME},
        f"/api/jobs/{ctl.CHALLENGE}/remaining_submissions/": {
            "participant_team": ctl.TEAM_NAME,
            "participant_team_id": ctl.TEAM,
            "phases": [{"id": ctl.PHASE, "limits": {
                "remaining_submissions_this_month_count": 5,
                "remaining_submissions_today_count": 1,
                "remaining_submissions_count": 10,
            }}],
        },
    }
    monkeypatch.setattr(ctl, "get_json", lambda path, token: replies[path])
    assert ctl.identity_and_quota("unused")["remaining_submissions_today_count"] == 1

    replies[f"/api/jobs/{ctl.CHALLENGE}/remaining_submissions/"]["phases"][0]["limits"]["remaining_submissions_today_count"] = 0
    with pytest.raises(RuntimeError, match="quota"):
        ctl.identity_and_quota("unused")


def test_identity_rejects_substring_impostor_and_wrong_authoritative_team(monkeypatch):
    quota = {
        "participant_team": ctl.TEAM_NAME,
        "participant_team_id": ctl.TEAM,
        "phases": [{"id": ctl.PHASE, "limits": {
            "remaining_submissions_this_month_count": 1,
            "remaining_submissions_today_count": 1,
            "remaining_submissions_count": 1,
        }}],
    }
    replies = {"/api/auth/user/": {"username": "x" + ctl.USERNAME}, f"/api/jobs/{ctl.CHALLENGE}/remaining_submissions/": quota}
    monkeypatch.setattr(ctl, "get_json", lambda path, token: replies[path])
    with pytest.raises(RuntimeError, match="account"):
        ctl.identity_and_quota("unused")
    replies["/api/auth/user/"] = {"username": ctl.USERNAME}
    quota["participant_team_id"] = 1
    with pytest.raises(RuntimeError, match="wrong team"):
        ctl.identity_and_quota("unused")
    quota["participant_team_id"] = ctl.TEAM
    quota["participant_team"] = "sustechhku-impostor"
    with pytest.raises(RuntimeError, match="wrong team"):
        ctl.identity_and_quota("unused")


def test_token_comparison_reads_each_path_once(monkeypatch):
    reads = []

    def fake_load(path):
        reads.append(path)
        return {"token": "same-token"}

    monkeypatch.setattr(ctl, "load_object", fake_load)
    assert ctl.bearer() == "same-token"
    assert reads == [ctl.REFERENCE_TOKEN, ctl.CURRENT_TOKEN]


def test_prior_uncertain_post_refuses_second_post(tmp_path):
    intent = tmp_path / "intent.json"
    audit = "a" * 64
    method = "method"
    uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/participant-team-42279:unique"
    ctl.save(intent, {"submitted_image_uri": uri, "candidate_audit_sha256": audit, "method_name": method, "post_attempted": True})
    with pytest.raises(RuntimeError, match="refuse a second POST"):
        ctl.reconcile_prior_intent(intent, [], tmp_path)

    valid = {
        "id": 11,
        "participant_team": ctl.TEAM,
        "challenge_phase": ctl.PHASE,
        "is_public": False,
        "method_name": method,
        "method_description": f"SSM private candidate image_uri={uri}; audit_manifest_sha256={audit}",
    }
    assert ctl.reconcile_prior_intent(intent, [valid], tmp_path) is True
    assert json.loads((tmp_path / "reconciled.json").read_text())["submission"]["id"] == 11
    duplicate = dict(valid, id=12)
    with pytest.raises(RuntimeError, match="ambiguous"):
        ctl.reconcile_prior_intent(intent, [valid, duplicate], tmp_path)


def test_save_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ctl.os, "fsync", lambda fd: calls.append(fd))
    ctl.save(tmp_path / "durable.json", {"ok": True})
    assert len(calls) == 2


def test_post_response_must_be_private_and_correct_scope():
    response = {"id": 3, "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "is_public": True}
    with pytest.raises(RuntimeError, match="private"):
        ctl.verify_post(response)
    response["is_public"] = False
    response["challenge_phase"] = 999
    with pytest.raises(RuntimeError, match="team or phase"):
        ctl.verify_post(response)


def test_ecr_binding_requires_exact_target_repository(monkeypatch):
    class FakeBoto:
        @staticmethod
        def client(*args, **kwargs):
            return {"client": args, "kwargs": kwargs}

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto)
    response = {
        "success": {
            "docker_repository_uri": f"123456789012.dkr.ecr.us-east-1.amazonaws.com/{ctl.ECR_REPOSITORY}",
            "federated_user": {
                "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "s", "SessionToken": "t"},
                "FederatedUser": {"FederatedUserId": "123456789012:user"},
            },
        }
    }
    monkeypatch.setattr(ctl, "get_json", lambda path, token: response)
    binding = ctl.ecr_binding("unused")
    assert binding["repository"].endswith(ctl.ECR_REPOSITORY)
    assert binding["account"] == "123456789012"

    response["success"]["docker_repository_uri"] = "123456789012.dkr.ecr.us-east-1.amazonaws.com/participant-team-1"
    with pytest.raises(RuntimeError, match="does not belong"):
        ctl.ecr_binding("unused")


def docker29_final_status(tag, digest, size=2501):
    return {"status": f"{tag}: digest: {digest} size: {size}"}


def image_manifest(config_digest, media_type="application/vnd.oci.image.manifest.v1+json"):
    encoded = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": media_type,
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": 702,
            },
            "layers": [],
        },
        separators=(",", ":"),
    )
    return encoded, "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest(), media_type


def install_push_fakes(
    monkeypatch,
    *,
    events,
    local_image_id,
    manifest_text,
    manifest_digest,
    manifest_media_type,
    descriptor=None,
    ecr_digest=None,
    batch_digest=None,
    reported_manifest_media_type=None,
):
    calls = {"push": [], "batch": []}

    class Image:
        id = local_image_id
        attrs = {} if descriptor is None else {"Descriptor": descriptor}

        def tag(self, repo, tag):
            calls["tag"] = (repo, tag)

    class Images:
        def get(self, tag):
            assert tag == "image"
            return Image()

        def push(self, repo, **kwargs):
            calls["push"].append((repo, kwargs))
            return iter(events)

    class DockerClient:
        images = Images()

        def login(self, **kwargs):
            calls["login"] = kwargs

    class DockerModule:
        @staticmethod
        def from_env():
            return DockerClient()

    class Ecr:
        def get_authorization_token(self, **kwargs):
            import base64

            return {
                "authorizationData": [
                    {
                        "authorizationToken": base64.b64encode(b"AWS:password").decode(),
                        "proxyEndpoint": "registry",
                    }
                ]
            }

        def describe_images(self, **kwargs):
            calls["describe"] = kwargs
            return {"imageDetails": [{"imageDigest": ecr_digest or manifest_digest}]}

        def batch_get_image(self, **kwargs):
            calls["batch"].append(kwargs)
            return {
                "images": [
                    {
                        "imageId": {
                            "imageTag": "tag",
                            "imageDigest": batch_digest or ecr_digest or manifest_digest,
                        },
                        "imageManifest": manifest_text,
                        "imageManifestMediaType": reported_manifest_media_type or manifest_media_type,
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "docker", DockerModule)
    return {"client": Ecr(), "account": "123"}, calls


def test_push_accepts_docker29_final_status_and_descriptor_manifest_identity(monkeypatch):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    descriptor = {"digest": manifest_digest, "mediaType": media_type, "size": len(manifest_text.encode("utf-8"))}
    binding, calls = install_push_fakes(
        monkeypatch,
        events=[
            {"status": "The push refers to repository [x/participant-team-42279]"},
            {"status": "layer: Pushed"},
            docker29_final_status("tag", manifest_digest),
            {"aux": {"Digest": manifest_digest, "Tag": "tag", "Size": 2501}},
        ],
        local_image_id=manifest_digest,
        manifest_text=manifest_text,
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
        descriptor=descriptor,
    )
    ctl.push_image("image", manifest_digest, "x/participant-team-42279:tag", binding)
    assert calls["push"] == [("x/participant-team-42279", {"tag": "tag", "stream": True, "decode": True})]
    assert calls["batch"] == [{
        "repositoryName": "participant-team-42279",
        "imageIds": [{"imageTag": "tag"}],
        "acceptedMediaTypes": [
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
        ],
    }]


def test_push_accepts_docker29_status_without_aux_for_classic_config_identity(monkeypatch):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", manifest_digest)],
        local_image_id=config_digest,
        manifest_text=manifest_text,
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
    )
    ctl.push_image("image", config_digest, "x/participant-team-42279:tag", binding)


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([{"error": "denied"}], "push failed"),
        ([{"errorDetail": {"message": "denied"}}], "push failed"),
        ([{"aux": {"Digest": "sha256:" + "a" * 64}}], "exactly one final"),
        ([{"status": "tag: digest: sha256:forged size: 1"}], "final tag/digest status is malformed"),
        ([docker29_final_status("other", "sha256:" + "a" * 64)], "final tag/digest status is malformed"),
        ([docker29_final_status("tag", "sha256:" + "a" * 64), {"aux": {}}], "auxiliary digest"),
        ([docker29_final_status("tag", "sha256:" + "a" * 64), {"aux": {"Digest": "sha256:" + "b" * 64}}], "status and auxiliary digest disagree"),
        ([docker29_final_status("tag", "sha256:" + "a" * 64), docker29_final_status("tag", "sha256:" + "b" * 64)], "exactly one final"),
    ],
)
def test_push_rejects_error_missing_forged_or_conflicting_docker29_events(monkeypatch, events, message):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    binding, _ = install_push_fakes(
        monkeypatch,
        events=events,
        local_image_id=config_digest,
        manifest_text=manifest_text,
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
    )
    with pytest.raises(RuntimeError, match=message):
        ctl.push_image("image", config_digest, "x/participant-team-42279:tag", binding)


def test_push_rejects_ecr_digest_and_classic_config_digest_mismatch(monkeypatch):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", manifest_digest)],
        local_image_id=config_digest,
        manifest_text=manifest_text,
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
        ecr_digest="sha256:" + "d" * 64,
    )
    with pytest.raises(RuntimeError, match="Docker push and ECR image digest disagree"):
        ctl.push_image("image", config_digest, "x/participant-team-42279:tag", binding)

    wrong_config_text, wrong_config_manifest, _ = image_manifest("sha256:" + "e" * 64)
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", wrong_config_manifest)],
        local_image_id=config_digest,
        manifest_text=wrong_config_text,
        manifest_digest=wrong_config_manifest,
        manifest_media_type=media_type,
    )
    with pytest.raises(RuntimeError, match="config digest differs"):
        ctl.push_image("image", config_digest, "x/participant-team-42279:tag", binding)


def test_push_rejects_ecr_manifest_bytes_digest_mismatch(monkeypatch):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    descriptor = {"digest": manifest_digest, "mediaType": media_type, "size": len(manifest_text.encode("utf-8"))}
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", manifest_digest)],
        local_image_id=manifest_digest,
        manifest_text=manifest_text + " ",
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
        descriptor=descriptor,
    )
    with pytest.raises(RuntimeError, match="manifest bytes digest differs"):
        ctl.push_image("image", manifest_digest, "x/participant-team-42279:tag", binding)


def test_push_rejects_manifest_body_response_media_type_and_schema_forgery(monkeypatch):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    descriptor = {"digest": manifest_digest, "mediaType": media_type, "size": len(manifest_text.encode("utf-8"))}
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", manifest_digest)],
        local_image_id=manifest_digest,
        manifest_text=manifest_text,
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
        reported_manifest_media_type="application/vnd.docker.distribution.manifest.v2+json",
        descriptor=descriptor,
    )
    with pytest.raises(RuntimeError, match="body and response media type disagree"):
        ctl.push_image("image", manifest_digest, "x/participant-team-42279:tag", binding)

    non_v2 = json.dumps(
        {
            "schemaVersion": 1,
            "mediaType": media_type,
            "config": {"digest": config_digest},
        },
        separators=(",", ":"),
    )
    non_v2_digest = "sha256:" + hashlib.sha256(non_v2.encode("utf-8")).hexdigest()
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", non_v2_digest)],
        local_image_id=non_v2_digest,
        manifest_text=non_v2,
        manifest_digest=non_v2_digest,
        manifest_media_type=media_type,
        descriptor={"digest": non_v2_digest, "mediaType": media_type, "size": len(non_v2.encode("utf-8"))},
    )
    with pytest.raises(RuntimeError, match="schema is not version 2"):
        ctl.push_image("image", non_v2_digest, "x/participant-team-42279:tag", binding)


@pytest.mark.parametrize(
    ("descriptor_change", "message"),
    [
        (lambda digest, media_type, size: {"digest": "sha256:" + "d" * 64, "mediaType": media_type, "size": size}, "Descriptor, ECR manifest, and audited image ID disagree"),
        (lambda digest, media_type, size: {"digest": digest, "mediaType": "application/vnd.docker.distribution.manifest.v2+json", "size": size}, "media type disagree"),
        (lambda digest, media_type, size: {"digest": digest, "mediaType": media_type, "size": size + 1}, "size disagree"),
    ],
)
def test_push_rejects_cross_forged_docker29_descriptor(monkeypatch, descriptor_change, message):
    config_digest = "sha256:" + "c" * 64
    manifest_text, manifest_digest, media_type = image_manifest(config_digest)
    descriptor = descriptor_change(manifest_digest, media_type, len(manifest_text.encode("utf-8")))
    binding, _ = install_push_fakes(
        monkeypatch,
        events=[docker29_final_status("tag", manifest_digest)],
        local_image_id=manifest_digest,
        manifest_text=manifest_text,
        manifest_digest=manifest_digest,
        manifest_media_type=media_type,
        descriptor=descriptor,
    )
    with pytest.raises(RuntimeError, match=message):
        ctl.push_image("image", manifest_digest, "x/participant-team-42279:tag", binding)


def test_detail_requires_private_scope_and_keeps_score_without_urls(monkeypatch):
    detail = {"id": 7, "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "is_public": False,
              "result": {"partition": "test_split_m1", "held_out_r2": 0.5, "nan": float("nan"), "nested": {"url": "https://secret.example/temporary", "token": "x", "held_in_r2": 0.4, "unsafe": "data:text/plain,x"}}, "submission_result_file": "https://secret.example/temporary"}
    monkeypatch.setattr(ctl, "get_json", lambda path, token: detail)
    safe = ctl.safe_detail(ctl.get_submission_detail("unused", 7))
    assert safe["result"] == {"partition": "test_split_m1", "held_out_r2": 0.5, "nested": {"held_in_r2": 0.4}}
    assert "submission_result_file" not in safe


def test_candidate_dedup_requires_canonical_audit_marker():
    audit = "b" * 64
    good = {"method_name": "m", "method_description": f"SSM private candidate image_uri=repo:one; audit_manifest_sha256={audit}"}
    same_audit_new_name = {"method_name": "renamed", "method_description": f"SSM private candidate image_uri=repo:two; audit_manifest_sha256={audit}"}
    wrong = {"method_name": "m", "method_description": f"SSM private candidate image_uri=repo:three; audit_manifest_sha256={'a' * 64}"}
    assert ctl.candidate_rows([good, same_audit_new_name, wrong], audit) == [good, same_audit_new_name]


def test_method_name_duplicate_gate_is_independent_of_audit_hash():
    rows = [{"id": 1, "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "method_name": "same"}]
    assert [row["id"] for row in rows if row.get("method_name") == "same"] == [1]
    assert ctl.candidate_rows(rows, "a" * 64) == []


def confirmed_active_intent(tmp_path, *, name="prior_intent.json", submission_id=73, method_name="prior method", audit_sha=None):
    audit_sha = audit_sha or "b" * 64
    uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/participant-team-42279:prior"
    state = {
        "schema": "ssm_evalai_private_intent_v1",
        "team": ctl.TEAM,
        "phase": ctl.PHASE,
        "post_attempted": True,
        "post_confirmed_utc": "2026-10-03T00:00:00+00:00",
        "submission_id": submission_id,
        "submitted_image_uri": uri,
        "candidate_audit_sha256": audit_sha,
        "method_name": method_name,
    }
    path = tmp_path / name
    write_json(path, state)
    row = {
        "id": submission_id,
        "status": "running",
        "is_public": False,
        "participant_team": ctl.TEAM,
        "challenge_phase": ctl.PHASE,
        "method_name": method_name,
        "method_description": f"SSM private candidate image_uri={uri}; audit_manifest_sha256={audit_sha}",
    }
    return path, state, row


def test_allow_active_intent_accepts_confirmed_distinct_real_shape_without_persisting_uri(tmp_path):
    path, _, row = confirmed_active_intent(tmp_path)
    allowed, receipt = ctl.allow_authenticated_active_intents(
        [path], [row], "current method", "a" * 64
    )
    assert allowed == [row]
    assert receipt == [{
        "intent_path": str(path.resolve()),
        "intent_sha256": ctl.sha256(path),
        "submission_id": 73,
    }]
    encoded = json.dumps(receipt)
    assert "image_uri" not in encoded
    assert "method_description" not in encoded
    assert "123456789012.dkr" not in encoded


@pytest.mark.parametrize(
    ("state_updates", "row_updates", "message"),
    [
        ({"schema": "forged"}, {}, "confirmed private"),
        ({"team": ctl.TEAM + 1}, {}, "confirmed private"),
        ({"team": True}, {}, "confirmed private"),
        ({"phase": ctl.PHASE + 1}, {}, "confirmed private"),
        ({"phase": True}, {}, "confirmed private"),
        ({"post_attempted": False}, {}, "confirmed private"),
        ({"post_confirmed_utc": ""}, {}, "confirmed private"),
        ({"post_confirmed_utc": None}, {}, "confirmed private"),
        ({"post_confirmed_utc": "forged"}, {}, "confirmed private"),
        ({"post_confirmed_utc": "2026-10-03T00:00:00"}, {}, "confirmed private"),
        ({"submission_id": True}, {}, "exact server reconciliation"),
        ({"submission_id": 0}, {}, "exact server reconciliation"),
        ({"submission_id": 74}, {}, "exactly one active"),
        ({}, {"is_public": True}, "active server row differ"),
        ({}, {"participant_team": ctl.TEAM + 1}, "active server row differ"),
        ({}, {"challenge_phase": ctl.PHASE + 1}, "active server row differ"),
        ({}, {"id": True}, "exactly one active"),
        ({}, {"method_name": "forged method"}, "active server row differ"),
        ({}, {"method_description": "SSM private candidate image_uri=forged; audit_manifest_sha256=" + "b" * 64}, "active server row differ"),
    ],
)
def test_allow_active_intent_rejects_forged_or_unconfirmed_bindings(tmp_path, state_updates, row_updates, message):
    path, state, row = confirmed_active_intent(tmp_path)
    state.update(state_updates)
    write_json(path, state)
    row.update(row_updates)
    with pytest.raises(RuntimeError, match=message):
        ctl.allow_authenticated_active_intents([path], [row], "current method", "a" * 64)


@pytest.mark.parametrize(
    ("method_name", "audit_sha"),
    [("current method", "b" * 64), ("prior method", "a" * 64)],
)
def test_allow_active_intent_rejects_current_method_or_candidate(tmp_path, method_name, audit_sha):
    path, _, row = confirmed_active_intent(tmp_path, method_name=method_name, audit_sha=audit_sha)
    with pytest.raises(RuntimeError, match="different method and candidate"):
        ctl.allow_authenticated_active_intents([path], [row], "current method", "a" * 64)


def test_allow_active_intent_requires_one_to_one_current_active_mapping(tmp_path):
    path, state, row = confirmed_active_intent(tmp_path)
    with pytest.raises(RuntimeError, match="path is repeated"):
        ctl.allow_authenticated_active_intents([path, path], [row], "current method", "a" * 64)

    copy = tmp_path / "identical_intent.json"
    copy.write_bytes(path.read_bytes())
    with pytest.raises(RuntimeError, match="content is repeated"):
        ctl.allow_authenticated_active_intents([path, copy], [row], "current method", "a" * 64)

    duplicate_row = dict(row, status="queued")
    with pytest.raises(RuntimeError, match="exactly one active"):
        ctl.allow_authenticated_active_intents([path], [row, duplicate_row], "current method", "a" * 64)

    second = dict(state, post_confirmed_utc="2026-10-03T00:01:00+00:00")
    second_path = tmp_path / "second_intent.json"
    write_json(second_path, second)
    with pytest.raises(RuntimeError, match="same submission ID"):
        ctl.allow_authenticated_active_intents([path, second_path], [row], "current method", "a" * 64)

    with pytest.raises(RuntimeError, match="exactly one active"):
        ctl.allow_authenticated_active_intents([path], [], "current method", "a" * 64)

    terminal = dict(row, status="finished")
    with pytest.raises(RuntimeError, match="active server row differ"):
        ctl.allow_authenticated_active_intents([path], [terminal], "current method", "a" * 64)


def install_main_preflight_fakes(tmp_path, monkeypatch, rows):
    manifest = {"method_name": "current method", "image": {"image_id": "sha256:" + "a" * 64}}
    candidate_manifest = tmp_path / "current_audit.json"
    candidate_manifest.write_text("current audit", encoding="utf-8")
    monkeypatch.setattr(ctl, "GLOBAL_LOCK", tmp_path / "global.lock")
    monkeypatch.setattr(ctl, "bearer", lambda: "token")
    monkeypatch.setattr(ctl, "audit_manifest", lambda *args: manifest.copy())
    monkeypatch.setattr(ctl, "validate_local_image", lambda *args: None)
    monkeypatch.setattr(ctl, "identity_and_quota", lambda token: {
        "remaining_submissions_this_month_count": 1,
        "remaining_submissions_today_count": 1,
        "remaining_submissions_count": 1,
    })
    monkeypatch.setattr(ctl, "list_submissions", lambda token: rows)
    return candidate_manifest


def test_main_active_intent_preflight_receipt_and_default_block(tmp_path, monkeypatch):
    path, _, row = confirmed_active_intent(tmp_path)
    candidate_manifest = install_main_preflight_fakes(tmp_path, monkeypatch, [row])
    receipt_dir = tmp_path / "receipt"
    base = ["controller", "--image", "image", "--task", "m1", "--candidate-manifest", str(candidate_manifest), "--receipt-dir", str(receipt_dir)]

    monkeypatch.setattr(sys, "argv", base + ["--preflight", "--allow-active-intent", str(path)])
    ctl.main()
    preflight = json.loads((receipt_dir / "preflight.json").read_text())
    assert preflight["active_submission_ids"] == [73]
    assert preflight["allowed_active_submission_ids"] == [73]
    assert preflight["blocked_active_submission_ids"] == []
    assert preflight["allowed_active_intents"] == [{
        "intent_path": str(path.resolve()), "intent_sha256": ctl.sha256(path), "submission_id": 73,
    }]
    encoded = json.dumps(preflight)
    assert "method_description" not in encoded
    assert "123456789012.dkr" not in encoded

    monkeypatch.setattr(sys, "argv", base + ["--execute"])
    with pytest.raises(RuntimeError, match="active or duplicate"):
        ctl.main()
    blocked = json.loads((receipt_dir / "preflight.json").read_text())
    assert blocked["allowed_active_submission_ids"] == []
    assert blocked["blocked_active_submission_ids"] == [73]


def test_main_accepts_repeatable_active_intent_paths_and_rejects_unused_terminal_path(tmp_path, monkeypatch):
    first_path, _, first = confirmed_active_intent(tmp_path, submission_id=73)
    second_path, _, second = confirmed_active_intent(
        tmp_path, name="second.json", submission_id=74, method_name="second prior method", audit_sha="c" * 64
    )
    candidate_manifest = install_main_preflight_fakes(tmp_path, monkeypatch, [first, second])
    receipt_dir = tmp_path / "receipt"
    base = ["controller", "--preflight", "--image", "image", "--task", "m1", "--candidate-manifest", str(candidate_manifest), "--receipt-dir", str(receipt_dir)]
    monkeypatch.setattr(sys, "argv", base + [
        "--allow-active-intent", str(first_path), "--allow-active-intent", str(second_path),
    ])
    ctl.main()
    preflight = json.loads((receipt_dir / "preflight.json").read_text())
    assert preflight["allowed_active_submission_ids"] == [73, 74]
    assert [entry["intent_sha256"] for entry in preflight["allowed_active_intents"]] == [
        ctl.sha256(first_path), ctl.sha256(second_path),
    ]

    terminal = dict(first, status="finished")
    terminal_receipt = tmp_path / "terminal_receipt"
    candidate_manifest = install_main_preflight_fakes(tmp_path, monkeypatch, [terminal])
    monkeypatch.setattr(sys, "argv", [
        "controller", "--preflight", "--image", "image", "--task", "m1", "--candidate-manifest", str(candidate_manifest),
        "--receipt-dir", str(terminal_receipt), "--allow-active-intent", str(first_path),
    ])
    with pytest.raises(RuntimeError, match="exactly one active"):
        ctl.main()


def test_main_execute_crosses_only_authenticated_active_intent_gate(tmp_path, monkeypatch):
    path, _, permitted = confirmed_active_intent(tmp_path)
    candidate_manifest = install_main_preflight_fakes(tmp_path, monkeypatch, [permitted])
    receipt_dir = tmp_path / "receipt"
    calls = []
    monkeypatch.setattr(ctl, "ecr_binding", lambda token: {
        "repository": "123456789012.dkr.ecr.us-east-1.amazonaws.com/participant-team-42279", "account": "123456789012",
    })
    monkeypatch.setattr(ctl, "push_image", lambda image, image_id, uri, binding: calls.append(("push", image, image_id, uri)))
    monkeypatch.setattr(ctl, "post_private", lambda token, manifest, uri, directory: calls.append(("post", uri)) or {
        "id": 91, "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "is_public": False, "status": "submitted",
    })
    monkeypatch.setattr(ctl, "poll_until_terminal", lambda token, submission_id, timeout: calls.append(("poll", submission_id)) or {
        "id": submission_id, "status": "finished", "is_public": False,
    })
    monkeypatch.setattr(sys, "argv", [
        "controller", "--execute", "--image", "image", "--task", "m1", "--candidate-manifest", str(candidate_manifest),
        "--receipt-dir", str(receipt_dir), "--allow-active-intent", str(path),
    ])
    ctl.main()
    assert [call[0] for call in calls] == ["push", "post", "poll"]
    intent = json.loads((receipt_dir / "intent.json").read_text())
    assert intent["post_attempted"] is True
    assert intent["submission_id"] == 91
    assert ctl.confirmed_utc(intent["post_confirmed_utc"])
    assert json.loads((receipt_dir / "terminal_91.json").read_text())["submission"]["status"] == "finished"


def test_main_keeps_current_candidate_and_method_duplicate_gates_after_active_allowance(tmp_path, monkeypatch):
    path, _, permitted = confirmed_active_intent(tmp_path)
    rows = [permitted]
    candidate_manifest = install_main_preflight_fakes(tmp_path, monkeypatch, rows)
    current_audit = ctl.sha256(candidate_manifest)
    receipt_dir = tmp_path / "receipt"
    base = [
        "controller", "--execute", "--image", "image", "--task", "m1", "--candidate-manifest", str(candidate_manifest),
        "--receipt-dir", str(receipt_dir), "--allow-active-intent", str(path),
    ]
    current_audit_duplicate = {
        "id": 90, "status": "finished", "is_public": False,
        "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "method_name": "old method",
        "method_description": f"SSM private candidate image_uri=repo:old; audit_manifest_sha256={current_audit}",
    }
    rows.append(current_audit_duplicate)
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(RuntimeError, match="active or duplicate"):
        ctl.main()
    preflight = json.loads((receipt_dir / "preflight.json").read_text())
    assert preflight["allowed_active_submission_ids"] == [73]
    assert preflight["candidate_audit_duplicate_submission_ids"] == [90]

    rows.pop()
    rows.append({
        "id": 92, "status": "finished", "is_public": False,
        "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "method_name": "current method",
        "method_description": "SSM private candidate image_uri=repo:old; audit_manifest_sha256=" + "c" * 64,
    })
    with pytest.raises(RuntimeError, match="active or duplicate"):
        ctl.main()
    preflight = json.loads((receipt_dir / "preflight.json").read_text())
    assert preflight["allowed_active_submission_ids"] == [73]
    assert preflight["method_name_duplicate_submission_ids"] == [92]


def test_main_blocks_mixture_when_only_one_active_row_is_explicitly_allowed(tmp_path, monkeypatch):
    path, _, permitted = confirmed_active_intent(tmp_path)
    unauthorized = dict(permitted, id=74, method_name="other prior method")
    unauthorized["method_description"] = unauthorized["method_description"].replace("prior", "other")
    candidate_manifest = install_main_preflight_fakes(tmp_path, monkeypatch, [permitted, unauthorized])
    receipt_dir = tmp_path / "receipt"
    monkeypatch.setattr(sys, "argv", [
        "controller", "--execute", "--image", "image", "--task", "m1", "--candidate-manifest", str(candidate_manifest),
        "--receipt-dir", str(receipt_dir), "--allow-active-intent", str(path),
    ])
    with pytest.raises(RuntimeError, match="active or duplicate"):
        ctl.main()
    preflight = json.loads((receipt_dir / "preflight.json").read_text())
    assert preflight["allowed_active_submission_ids"] == [73]
    assert preflight["blocked_active_submission_ids"] == [74]


def test_main_negative_gates_never_reach_push_or_post(tmp_path, monkeypatch):
    manifest = {"method_name": "m", "image": {"image_id": "sha256:" + "a" * 64}}
    calls = []
    monkeypatch.setattr(ctl, "GLOBAL_LOCK", tmp_path / "global.lock")
    monkeypatch.setattr(ctl, "bearer", lambda: "token")
    monkeypatch.setattr(ctl, "audit_manifest", lambda *args: manifest.copy())
    monkeypatch.setattr(ctl, "sha256", lambda path: "b" * 64)
    monkeypatch.setattr(ctl, "validate_local_image", lambda *args: None)
    monkeypatch.setattr(ctl, "identity_and_quota", lambda token: {"remaining_submissions_this_month_count": 1, "remaining_submissions_today_count": 1, "remaining_submissions_count": 1})
    monkeypatch.setattr(ctl, "push_image", lambda *args: calls.append("push"))
    monkeypatch.setattr(ctl, "post_private", lambda *args: calls.append("post"))
    monkeypatch.setattr(sys, "argv", ["controller", "--execute", "--image", "image", "--task", "m1", "--candidate-manifest", "audit", "--receipt-dir", str(tmp_path)])

    uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/participant-team-42279:unknown"
    ctl.save(tmp_path / "intent.json", {"submitted_image_uri": uri, "candidate_audit_sha256": "b" * 64, "method_name": "m", "post_attempted": True})
    monkeypatch.setattr(ctl, "list_submissions", lambda token: [])
    with pytest.raises(RuntimeError, match="refuse a second POST"):
        ctl.main()
    assert calls == []

    (tmp_path / "intent.json").unlink()
    monkeypatch.setattr(ctl, "audit_manifest", lambda *args: (_ for _ in ()).throw(RuntimeError("artifact bad")))
    with pytest.raises(RuntimeError, match="artifact bad"):
        ctl.main()
    monkeypatch.setattr(ctl, "audit_manifest", lambda *args: manifest.copy())
    monkeypatch.setattr(ctl, "validate_local_image", lambda *args: (_ for _ in ()).throw(RuntimeError("image bad")))
    with pytest.raises(RuntimeError, match="image bad"):
        ctl.main()
    monkeypatch.setattr(ctl, "validate_local_image", lambda *args: None)
    monkeypatch.setattr(ctl, "identity_and_quota", lambda *args: (_ for _ in ()).throw(RuntimeError("quota bad")))
    with pytest.raises(RuntimeError, match="quota bad"):
        ctl.main()
    monkeypatch.setattr(ctl, "identity_and_quota", lambda token: {"remaining_submissions_this_month_count": 1, "remaining_submissions_today_count": 1, "remaining_submissions_count": 1})
    monkeypatch.setattr(ctl, "list_submissions", lambda token: [{"id": 9, "participant_team": ctl.TEAM, "challenge_phase": ctl.PHASE, "status": "running"}])
    with pytest.raises(RuntimeError, match="active"):
        ctl.main()
    assert calls == []
