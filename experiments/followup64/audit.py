"""GDM64 exact-attempt operation-only reproducibility audit driver.

This intentionally reuses the bounded-resource GDM63 audit implementation and transforms
only the experiment-specific provenance and representation assertions under exact
replacement-count checks. It never trains a model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

SOURCE_RUN = 35995968508
SOURCE_SHA = "6fe0f89630324fa18c18175c1d7c64ba614a1cff"

ATTEMPTS = {
    1: {
        "preflight_job": 107620842012,
        "operation_job": 107621359661,
        "report_job": 107628882108,
        "preflight_artifact": 10806327006,
        "preflight_digest": "sha256:3a4a8fd3e45fdf6d2b144ee34c37c5bfb1b5262c3857442c0587227676605044",
        "operation_artifact": 10806744697,
        "operation_digest": "sha256:b82bdcecd8287e40f3464b5abed355baae131e93b5fc13d875f62ebdf445d2ac",
        "report_artifact": 10806439103,
        "report_digest": "sha256:7265c0c821c2b5d483efdcf1dc15ba7bdfd4d7eccd22dfdfaa1ff31a69667a05",
    },
    2: {
        "preflight_job": 107635343662,
        "operation_job": 107636115727,
        "report_job": 107643957581,
        "preflight_artifact": 10808385906,
        "preflight_digest": "sha256:eaa0b7a3e3ddf8eae40d2ecfa02d395f4ab4d7c9ed9853b459210827f7378e6e",
        "operation_artifact": 10809460402,
        "operation_digest": "sha256:cf81d28921824ce9b15aad4ee3c89d00bb3d38fb1297a2c86343b0a099ef50da",
        "report_artifact": 10809186317,
        "report_digest": "sha256:701f7edef0c7f521c4b60781e6ce4866fc58b1d109ca90d8fabab34b2b913d2c",
    },
}


def replace_once(script: str, old: str, new: str) -> str:
    count = script.count(old)
    if count != 1:
        raise AssertionError((count, old[:160]))
    return script.replace(old, new, 1)


def extract_template() -> str:
    template = Path(".github/workflows/gdm63-operation-reproducibility-audit.yml").read_text()
    start_marker = "          python -u - <<'PY' 2>&1 | tee audit-output/audit.log\n"
    end_marker = "\n          PY\n"
    if template.count(start_marker) != 1:
        raise AssertionError("unexpected GDM63 audit template start marker count")
    start = template.index(start_marker) + len(start_marker)
    end = template.index(end_marker, start)
    return textwrap.dedent(template[start:end])


def build_script() -> str:
    script = extract_template()

    script = replace_once(
        script,
        """from experiments.followup63.collect import verify_summary
from experiments.followup63.run import (
    ARMS,
    INTERACTION_MODES,
    LATENT_WIDTH,
    TOPK,
    ambiguity_feature_dim,
    interaction_parameter_count,
    _interaction_description,
    _interaction_state_hash,
)
""",
        """from experiments.followup64.collect import verify_summary
from experiments.followup64.run import (
    ARMS,
    REPRESENTATION_MODES,
    BLOCK_KINDS,
    TOPK,
    ambiguity_feature_dim,
    _representation_description,
    CANONICAL_GDM63_SOURCE,
    CANONICAL_GDM63_AUDIT_RUN,
    CANONICAL_GDM63_PROMOTION,
)
""",
    )

    script = replace_once(
        script,
        """SEEDS = (6301, 6302)
TASK = 'operation'
ARM_NAMES = tuple(ARMS)
require_arms = {
    'learned-local-control',
    'learned-top1-cross',
    'learned-neighbor-cross',
    'learned-allpairs-cross',
    'learned-competitive-cross',
}
if set(ARM_NAMES) != require_arms or set(INTERACTION_MODES) != require_arms or len(ARM_NAMES) != 5:
    raise AssertionError('unexpected GDM63 arm contract')
""",
        """SEEDS = (6401, 6402)
TASK = 'operation'
ARM_NAMES = tuple(ARMS)
require_arms = {
    'whole-request-control',
    'parent-request-factor',
    'leaf-request-factor',
    'split-parent-leaf-product',
    'split-parent-leaf-delta',
}
if set(ARM_NAMES) != require_arms or set(REPRESENTATION_MODES) != require_arms or len(ARM_NAMES) != 5:
    raise AssertionError('unexpected GDM64 arm contract')
assert BLOCK_KINDS == {
    'whole-request-control': ('whole-product', 'whole-delta'),
    'parent-request-factor': ('parent-product', 'parent-delta'),
    'leaf-request-factor': ('leaf-product', 'leaf-delta'),
    'split-parent-leaf-product': ('parent-product', 'leaf-product'),
    'split-parent-leaf-delta': ('parent-delta', 'leaf-delta'),
}
""",
    )

    start = script.index("ATTEMPTS = {")
    end = script.index("\n\nreport = {", start)
    attempts_literal = "ATTEMPTS = " + repr(ATTEMPTS).replace("}, ", "},\n    ")
    script = script[:start] + attempts_literal + script[end:]

    script = replace_once(
        script,
        "'format': 'gdm63-operation-reproducibility-audit-v1'",
        "'format': 'gdm64-operation-reproducibility-audit-v1'",
    )
    script = replace_once(script, "'learned-allpairs-cross' in s", "'split-parent-leaf-product' in s")
    script = replace_once(script, "'wrong GDM63 smoke arm evidence'", "'wrong GDM64 smoke arm evidence'")
    script = replace_once(
        script,
        "summary['config']['arm'] == 'learned-allpairs-cross'",
        "summary['config']['arm'] == 'split-parent-leaf-product'",
    )
    script = replace_once(script, "int(summary['seed']) == 6301", "int(summary['seed']) == 6401")
    script = replace_once(
        script,
        "    require(_interaction_state_hash(state) == training[f'gdm63_{phase}_interaction_hash'], f'{phase} interaction hash mismatch')\n",
        "",
    )

    script = script.replace("gdm63-operation-batch-report-v1", "gdm64-operation-batch-report-v1")
    script = script.replace("experiments.followup63.collect", "experiments.followup64.collect")
    script = script.replace("'6301,6302'", "'6401,6402'")
    script = script.replace("schema evidence leaked into GDM63 source report", "schema evidence leaked into GDM64 source report")

    script = replace_once(
        script,
        "              adapter_counts = set()\n              ambiguity_counts = set()\n",
        "              ambiguity_counts = set()\n",
    )

    old_validation = """                  require(training.get('gdm63_family') == 'learned-rank-preserving-cross-candidate-interaction', 'missing GDM63 family receipt')
                  require(training.get('gdm63_interaction_mode') == arm, 'interaction mode receipt mismatch')
                  require(training.get('gdm63_requested_topk') == TOPK == 5, 'wrong TOP5 receipt')
                  require(training.get('gdm63_rank_preserving') is True, 'rank preservation missing')
                  require(training.get('gdm63_raw_representation') == 'canonical ranked request-product q*ci plus request-delta abs(q-ci)', 'raw representation changed')
                  require(training.get('gdm63_interaction_description') == _interaction_description(arm), 'interaction description mismatch')
                  require(training.get('gdm63_latent_width') == LATENT_WIDTH == 4, 'latent width mismatch')
                  expected_dim = ambiguity_feature_dim(768)
                  require(training.get('gdm63_ambiguity_feature_dim') == expected_dim == 7691, 'raw ambiguity dimension mismatch')
                  require(training.get('gdm63_interaction_parameter_count') == interaction_parameter_count(768), 'interaction parameter count mismatch')
                  adapter_counts.add(training.get('gdm63_interaction_parameter_count'))
                  ambiguity_counts.add(training.get('gdm63_ambiguity_parameter_count'))
                  changed = training.get('gdm63_changed_interaction_parameter_tensors')
                  require(isinstance(changed, list) and 'ambiguity.proj.weight' in changed, 'interaction projection did not change')
                  require(training.get('gdm56_ambiguity_curriculum') == 'family-balanced', 'curriculum changed')
                  require(float(training.get('gdm56_counterfactual_negative_fraction', -1)) == 0.0, 'counterfactual curriculum changed')
                  validate_checkpoint(summary, directory, 'initial')
                  validate_checkpoint(summary, directory, 'selected')
                  require(training['initial_state_hash'] != training['selected_state_hash'], 'full learned state did not change')
                  require(training['initial_capability_hash'] != training['selected_capability_hash'], 'capability head did not change')
                  require(training['initial_ambiguity_hash'] != training['selected_ambiguity_hash'], 'ambiguity head did not change')
                  require(training['gdm63_initial_interaction_hash'] != training['gdm63_selected_interaction_hash'], 'interaction adapter did not change')
"""
    new_validation = """                  require(training.get('gdm64_family') == 'schema-coordinate-semantic-factorization', 'missing GDM64 family receipt')
                  require(training.get('gdm64_representation_mode') == arm, 'representation mode receipt mismatch')
                  require(training.get('gdm64_requested_topk') == TOPK == 5, 'wrong TOP5 receipt')
                  require(training.get('gdm64_rank_preserving') is True, 'rank preservation missing')
                  expected_dim = ambiguity_feature_dim(768)
                  require(training.get('gdm64_ambiguity_feature_dim') == expected_dim == 7691, 'ambiguity dimension mismatch')
                  require(training.get('gdm64_semantic_representation') == _representation_description(arm), 'representation description mismatch')
                  first, second = BLOCK_KINDS[arm]
                  require(training.get('gdm64_first_block') == first, 'first semantic block mismatch')
                  require(training.get('gdm64_second_block') == second, 'second semantic block mismatch')
                  require(training.get('gdm64_parent_text_contract') == 'GraphQL parent path: + path[1:-1] joined by dot; <root> if empty', 'parent text contract mismatch')
                  require(training.get('gdm64_leaf_text_contract') == 'GraphQL leaf field: + path[-1]', 'leaf text contract mismatch')
                  require(training.get('gdm64_catalog_semantic_views_cached') is True, 'catalog semantic views must be cached')
                  require(training.get('canonical_gdm63_source_commit') == CANONICAL_GDM63_SOURCE, 'canonical GDM63 source receipt mismatch')
                  require(training.get('canonical_gdm63_audit_run') == CANONICAL_GDM63_AUDIT_RUN, 'canonical GDM63 audit receipt mismatch')
                  require(training.get('canonical_gdm63_promotion_commit') == CANONICAL_GDM63_PROMOTION, 'canonical GDM63 promotion receipt mismatch')
                  ambiguity_counts.add(training.get('gdm64_ambiguity_parameter_count'))
                  require(training.get('gdm56_ambiguity_curriculum') == 'family-balanced', 'curriculum changed')
                  require(float(training.get('gdm56_counterfactual_negative_fraction', -1)) == 0.0, 'counterfactual curriculum changed')
                  validate_checkpoint(summary, directory, 'initial')
                  validate_checkpoint(summary, directory, 'selected')
                  require(training['initial_state_hash'] != training['selected_state_hash'], 'full learned state did not change')
                  require(training['initial_capability_hash'] != training['selected_capability_hash'], 'capability head did not change')
                  require(training['initial_ambiguity_hash'] != training['selected_ambiguity_hash'], 'ambiguity head did not change')
"""
    script = replace_once(script, old_validation, new_validation)
    script = replace_once(
        script,
        "require(len(adapter_counts) == 1 and len(ambiguity_counts) == 1, f'matched-capacity contract broken attempt {attempt}')",
        "require(len(ambiguity_counts) == 1, f'matched-capacity ambiguity-head contract broken attempt {attempt}')",
    )

    # The report's audit_workflow_commit comes from GITHUB_SHA; source evidence remains pinned separately.
    return script


def main() -> None:
    if os.environ.get("SOURCE_RUN") and int(os.environ["SOURCE_RUN"]) != SOURCE_RUN:
        raise AssertionError("SOURCE_RUN environment mismatch")
    if os.environ.get("SOURCE_SHA") and os.environ["SOURCE_SHA"] != SOURCE_SHA:
        raise AssertionError("SOURCE_SHA environment mismatch")

    script = build_script()
    generated = Path("audit-script.py")
    generated.write_text(script)
    subprocess.run([sys.executable, "-m", "py_compile", str(generated)], check=True)
    subprocess.run([sys.executable, "-u", str(generated)], check=True)

    report_path = Path("audit-output/REPRODUCIBILITY_REPORT.json")
    if not report_path.is_file():
        raise AssertionError("missing reproducibility report")
    report = json.loads(report_path.read_text())
    assert report["format"] == "gdm64-operation-reproducibility-audit-v1"
    assert report["complete"] is True
    assert report["evidence_verified"] is True
    assert report["reproducible"] is True
    assert report["promotion_eligible"] is True
    assert report["errors"] == []
    assert report["expected_operation_configs"] == 10
    assert report["matched_operation_configs"] == 10
    assert report["schema_configs_compared"] == 0
    assert len(report["comparisons"]) == 10
    assert all(x["exact_match"] and x["differing_fields"] == [] for x in report["comparisons"])
    print("GDM64 operation-only exact-attempt audit passed 10/10; schema evidence excluded")


if __name__ == "__main__":
    main()
