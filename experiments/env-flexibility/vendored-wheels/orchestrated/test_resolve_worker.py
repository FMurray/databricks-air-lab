import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("air_wheelhouse_resolver", HERE / "resolve_worker.py")
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


class ResolverLogicTest(unittest.TestCase):
    def test_delta_compares_normalized_names_and_versions(self):
        closure = {"same": "1.0", "changed": "2.0", "new-package": "3.0"}
        baseline = {"same": "1.0.0", "changed": "1.0"}

        self.assertEqual(
            resolver._version_delta(closure, baseline),
            {"changed": "2.0", "new-package": "3.0"},
        )

    def test_direct_resolution_rejects_missing_or_wrong_direct_packages(self):
        direct, issues = resolver._direct_resolution(
            ["transformers==5.15.0", "cdaosdk-openai==2.3.2"],
            {"transformers": "4.57.1"},
            {"transformers": "4.57.1"},
            {},
        )

        self.assertEqual(
            direct,
            ["transformers==5.15.0", "cdaosdk-openai==2.3.2"],
        )
        self.assertEqual(
            [issue["requirement"] for issue in issues],
            ["transformers==5.15.0", "cdaosdk-openai==2.3.2"],
        )

    def test_direct_resolution_labels_new_and_changed_packages_as_delta(self):
        direct, issues = resolver._direct_resolution(
            ["transformers==5.15.0", "cdaosdk-openai==2.3.2"],
            {"transformers": "5.15.0", "cdaosdk-openai": "2.3.2"},
            {"transformers": "4.57.1"},
            {},
        )

        self.assertEqual(issues, [])
        self.assertEqual(
            direct,
            [
                "transformers==5.15.0 -> delta transformers==5.15.0",
                "cdaosdk-openai==2.3.2 -> delta cdaosdk-openai==2.3.2",
            ],
        )

    def test_uv_platform_uses_the_highest_supported_manylinux_target(self):
        target = {
            "platforms": ["manylinux_2_39_x86_64", "manylinux_2_17_x86_64"],
            "marker_environment": {"sys_platform": "linux", "platform_machine": "x86_64"},
        }

        self.assertEqual(resolver._uv_platform(target), "x86_64-manylinux_2_39")

    def test_supported_tags_include_native_abi3_and_pure_python_wheels(self):
        target = {
            "python_version": "312",
            "implementation": "cp",
            "abi": "cp312",
            "abis": ["cp312", "abi3", "none"],
            "platforms": ["manylinux_2_39_x86_64", "manylinux_2_17_x86_64"],
        }

        tags = resolver._supported_target_tags(target)

        self.assertIn("cp312-cp312-manylinux_2_39_x86_64", tags)
        self.assertIn("cp312-abi3-manylinux_2_17_x86_64", tags)
        self.assertIn("py3-none-any", tags)

    def test_environment_yaml_replays_the_verified_offline_install(self):
        rendered = resolver._environment_yaml(
            {
                "air_environment": "databricks_ai_v5",
                "environment_version": "5",
            },
            Path("/Volumes/catalog/schema/wheels/builds/abc/wheelhouse"),
            Path("/Volumes/catalog/schema/wheels/builds/abc/environment.lock"),
            has_delta=True,
        )

        self.assertEqual(
            rendered,
            'base_environment: "databricks_ai_v5"\n'
            'environment_version: "5"\n'
            "dependencies:\n"
            '  - "--no-index"\n'
            '  - "--require-hashes"\n'
            '  - "--find-links /Volumes/catalog/schema/wheels/builds/abc/wheelhouse"\n'
            '  - "-r /Volumes/catalog/schema/wheels/builds/abc/environment.lock"\n',
        )

    def test_checked_in_v5_profile_is_a_complete_exact_pin_set(self):
        profile = HERE / "profiles" / "databricks_ai_v5"
        target = json.loads((profile / "target_env.json").read_text())
        pins = resolver._load_pins(profile / "constraints.txt")

        self.assertEqual(target["profile_schema"], 1)
        self.assertEqual(target["air_environment"], "databricks_ai_v5")
        self.assertEqual(target["python_full"], "3.12.3")
        self.assertEqual(target["top_tag"], "cp312-cp312-manylinux_2_39_x86_64")
        self.assertEqual(target["source"]["run_id"], "52417241507965")
        self.assertEqual(len(pins), 433)


if __name__ == "__main__":
    unittest.main()
