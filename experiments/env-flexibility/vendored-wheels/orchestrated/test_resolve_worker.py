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

    def test_environment_yaml_lists_exact_wheels_without_an_index_or_include(self):
        rendered = resolver._environment_yaml(
            {
                "air_environment": "databricks_ai_v5",
                "environment_version": "5",
            },
            [
                Path("/Volumes/catalog/schema/wheels/builds/abc/wheelhouse/one-1-py3-none-any.whl"),
                Path("/Volumes/catalog/schema/wheels/builds/abc/wheelhouse/two-2-py3-none-any.whl"),
            ],
        )

        self.assertEqual(
            rendered,
            'base_environment: "databricks_ai_v5"\n'
            'environment_version: "5"\n'
            "dependencies:\n"
            '  - "--no-index"\n'
            '  - "/Volumes/catalog/schema/wheels/builds/abc/wheelhouse/one-1-py3-none-any.whl"\n'
            '  - "/Volumes/catalog/schema/wheels/builds/abc/wheelhouse/two-2-py3-none-any.whl"\n',
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
