"""Keep the public release independent of the private training implementations.

Optional Hugging Face forward-reference tests are inference checks, not an
alternative training implementation. They remain available for JAX validation.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MODULES = {"expdis_torch"}


class JaxReleaseBoundaryTests(unittest.TestCase):
    def test_private_training_implementations_are_absent(self):
        forbidden = (*sorted(PRIVATE_MODULES), "run_expdis_torch.py")
        present = [name for name in forbidden if (ROOT / name).exists()]
        self.assertEqual(present, [], f"Private training source in release: {present}")

    def test_retired_pilot_campaign_roots_are_absent(self):
        present = [name for name in ("scripts", "configs") if (ROOT / name).exists()]
        self.assertEqual(present, [], f"Pilot campaign files in release: {present}")

    def test_jax_does_not_import_private_training_modules(self):
        violations = []
        for path in sorted((ROOT / "expdis_jax").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in PRIVATE_MODULES:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
