import importlib.util
import unittest
from pathlib import Path


WORKER = Path(__file__).with_name("resolve_worker.py")
SPEC = importlib.util.spec_from_file_location("air_wheelhouse_resolver", WORKER)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def report_item(name, version, requires=()):
    return {
        "metadata": {
            "name": name,
            "version": version,
            "requires_dist": list(requires),
        }
    }


class ResolverLogicTest(unittest.TestCase):
    def test_detects_transitive_version_conflict_without_upgrading_compatible_baseline(self):
        baseline = {"openai": "1.0.0", "jiter": "0.8.0", "numpy": "2.1.3"}
        wants = [
            report_item("openai", "2.0.0", ["jiter>=0.10,<1", "numpy>=2"]),
            report_item("jiter", "0.11.0"),
            report_item("numpy", "2.3.0"),
        ]

        result = resolver.infer_required_overrides(
            ["openai==2.0.0"], wants, baseline, marker_environment={}
        )

        self.assertEqual(result["overrides"], {"openai", "jiter"})
        self.assertNotIn("numpy", result["overrides"])

    def test_reuses_compatible_baseline_parent_instead_of_following_newer_candidate(self):
        baseline = {"parent": "1.0.0", "shared-dependency": "1.0.0"}
        wants = [
            report_item("parent", "2.0.0", ["shared-dependency>=2"]),
            report_item("shared-dependency", "2.0.0"),
        ]

        result = resolver.infer_required_overrides(
            ["parent>=1"], wants, baseline, marker_environment={}
        )

        self.assertEqual(result["overrides"], set())

    def test_delta_compares_normalized_versions_as_well_as_names(self):
        closure = {"same": "1.0", "changed": "2.0", "new-package": "3.0"}
        baseline = {"same": "1.0.0", "changed": "1.0"}

        self.assertEqual(
            resolver._version_delta(closure, baseline),
            {"changed": "2.0", "new-package": "3.0"},
        )

    def test_wheel_requires_a_tag_from_air_supported_tags(self):
        wheel = "jiter-0.11.0-cp312-cp312-manylinux_2_17_x86_64.whl"

        self.assertEqual(
            resolver._compatible_wheel_tags(
                wheel, {"cp312-cp312-manylinux_2_17_x86_64"}
            ),
            ["cp312-cp312-manylinux_2_17_x86_64"],
        )
        self.assertEqual(
            resolver._compatible_wheel_tags(wheel, {"cp312-cp312-macosx_14_0_arm64"}),
            [],
        )

    def test_target_args_include_every_air_abi_and_platform(self):
        args = resolver._target_args({
            "implementation": "cp",
            "python_version": "312",
            "abi": "cp312",
            "abis": ["cp312", "abi3", "none"],
            "platforms": ["manylinux_2_39_x86_64", "manylinux_2_17_x86_64"],
        })

        self.assertEqual(args.count("--abi"), 3)
        self.assertEqual(args.count("--platform"), 2)

    def test_restores_compatible_pins_and_keeps_hidden_parent_relaxed(self):
        def can_resolve(overrides):
            # The target graph works only when parent is allowed to move. Array is compatible and
            # should be restored; shared was a directly proven conflict and remains fixed.
            return "parent" in overrides

        overrides, restored, probes = resolver.restore_compatible_pins(
            {"array", "parent", "shared"},
            {"shared"},
            can_resolve,
        )

        self.assertEqual(overrides, {"parent", "shared"})
        self.assertEqual(restored, {"array"})
        self.assertGreater(probes, 0)


if __name__ == "__main__":
    unittest.main()
