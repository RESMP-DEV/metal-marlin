#!/usr/bin/env python3
"""Generate API documentation for metal_marlin using pdoc3.

This script generates HTML API documentation without importing the module,
avoiding torch initialization issues during doc generation.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent
    docs_api = repo_root / "docs" / "api"
    docs_api.mkdir(parents=True, exist_ok=True)
    
    # Generate docs without importing (skip errors mode)
    cmd = [
        sys.executable, "-m", "pdoc",
        "--html",
        "--output-dir", str(docs_api),
        "--force",
        "--skip-errors",  # Skip import errors
        "metal_marlin"
    ]
    
    env = os.environ.copy()
    # Prevent torch from loading heavy resources during doc generation
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["TORCH_FORCE_WHEELS"] = "1"
    
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating docs:\n{result.stderr}", file=sys.stderr)
        # Try alternative: generate stubs only
        print("\nFalling back to stub generation...", file=sys.stderr)
        generate_stubs(docs_api)
    else:
        print(f"✓ API docs generated at: {docs_api}")
        if result.stdout:
            print(result.stdout)

def generate_stubs(docs_api: Path):
    """Generate minimal API documentation from source analysis."""
    import ast
    from pathlib import Path
    
    repo_root = Path(__file__).parent.parent
    metal_marlin = repo_root / "metal_marlin"
    
    # Create index
    index_content = """# Metal Marlin API Reference

## Core Modules

"""
    
    py_files = sorted(metal_marlin.glob("*.py"))
    for py_file in py_files:
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        
        module_name = py_file.stem
        index_content += f"- [{module_name}]({module_name}.html)\n"
        
        # Parse and extract docstrings
        try:
            with open(py_file) as f:
                tree = ast.parse(f.read())
            
            module_doc = ast.get_docstring(tree) or "No module docstring"
            
            # Extract classes and functions
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    classes.append((node.name, doc))
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("_"):
                        doc = ast.get_docstring(node) or ""
                        # Get signature
                        args = [arg.arg for arg in node.args.args]
                        sig = f"{node.name}({', '.join(args)})"
                        functions.append((sig, doc))
            
            # Generate module page
            module_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{module_name} - Metal Marlin</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ margin-top: 30px; color: #555; }}
        .item {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
        .signature {{ font-family: Monaco, monospace; color: #0066cc; }}
        pre {{ background: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{module_name}</h1>
    <p>{module_doc}</p>
"""
            
            if classes:
                module_html += "\n<h2>Classes</h2>\n"
                for name, doc in classes:
                    module_html += f'<div class="item"><strong class="signature">{name}</strong><p>{doc}</p></div>\n'
            
            if functions:
                module_html += "\n<h2>Functions</h2>\n"
                for sig, doc in functions:
                    module_html += f'<div class="item"><code class="signature">{sig}</code><p>{doc}</p></div>\n'
            
            module_html += "</body></html>"
            
            (docs_api / f"{module_name}.html").write_text(module_html)
            
        except Exception as e:
            print(f"Warning: Could not parse {py_file.name}: {e}")
    
    # Write index
    index_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Metal Marlin API Reference</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
        ul { line-height: 1.8; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Metal Marlin API Reference</h1>
    <p>FP4/INT4 quantization for LLMs on Apple Silicon</p>
    <h2>Core Modules</h2>
    <ul>
"""
    
    for py_file in py_files:
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        module_name = py_file.stem
        index_html += f'        <li><a href="{module_name}.html">{module_name}</a></li>\n'
    
    index_html += """    </ul>
</body>
</html>"""
    
    (docs_api / "index.html").write_text(index_html)
    print(f"✓ Stub API docs generated at: {docs_api}")

if __name__ == "__main__":
    main()
