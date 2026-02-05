"""Slither static analysis integration for P6 tool-augmented prompts."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set


def _get_slither_executable() -> str:
    """Get the path to slither executable, preferring venv's version."""
    # First, try the venv's bin directory (same as current Python interpreter)
    venv_slither = Path(sys.executable).parent / "slither"
    if venv_slither.exists():
        return str(venv_slither)
    # Fall back to PATH lookup
    return "slither"


@dataclass
class SlitherFinding:
    """A single finding from Slither."""
    check: str
    impact: str
    confidence: str
    description: str
    elements: List[Dict] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format finding for inclusion in prompt."""
        lines = []
        lines.append(f"**{self.check}** [{self.impact.upper()} Impact, {self.confidence.upper()} Confidence]")
        lines.append(f"  {self.description}")

        # Extract relevant code locations
        for elem in self.elements[:3]:  # Limit to avoid prompt bloat
            if elem.get("source_mapping"):
                src = elem["source_mapping"]
                if src.get("lines"):
                    lines.append(f"  - Lines: {src['lines'][0]}-{src['lines'][-1] if len(src['lines']) > 1 else src['lines'][0]}")

        return "\n".join(lines)

    def get_affected_functions(self) -> List[str]:
        """Get function names affected by this finding.

        Returns:
            List of function names extracted from elements
        """
        return [
            elem.get("name")
            for elem in self.elements
            if elem.get("type") == "function" and elem.get("name")
        ]

    def get_line_range(self) -> Optional[Tuple[int, int]]:
        """Get (start_line, end_line) for this finding.

        Returns:
            Tuple of (min_line, max_line) or None if no line info
        """
        for elem in self.elements:
            if elem.get("source_mapping", {}).get("lines"):
                lines = elem["source_mapping"]["lines"]
                return (min(lines), max(lines))
        return None


@dataclass
class SlitherResult:
    """Result of running Slither on a contract."""
    success: bool
    findings: List[SlitherFinding]
    detector_count: int
    error: Optional[str] = None
    raw_output: Optional[str] = None
    slither: Optional["Slither"] = None  # Optional Slither instance for deep analysis
    _source: Optional[str] = None  # Source code for lazy Slither creation

    def get_slither_instance(self) -> Optional["Slither"]:
        """
        Get or create Slither instance for deep analysis.

        Lazily creates a Slither object when first requested.
        This is needed for static confirmation modules (DF, VC, OC, FA).

        Returns:
            Slither instance or None if creation fails
        """
        if self.slither is not None:
            return self.slither

        if not self._source or not self.success:
            return None

        try:
            from slither.slither import Slither
            import tempfile

            # Create temp file for Slither
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.sol', delete=False
            ) as f:
                f.write(self._source)
                temp_path = f.name

            try:
                self.slither = Slither(temp_path)
                return self.slither
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        except ImportError:
            # slither-analyzer not installed with Python API
            return None
        except Exception:
            # Slither creation failed
            return None

    def to_prompt_text(self) -> str:
        """Format all findings for inclusion in prompt."""
        if not self.success:
            return f"Slither analysis failed: {self.error}"

        if not self.findings:
            return "No issues detected by Slither."

        lines = [f"Slither detected {len(self.findings)} potential issues:\n"]

        # Group by impact
        by_impact = {"high": [], "medium": [], "low": [], "informational": []}
        for f in self.findings:
            impact = f.impact.lower()
            if impact in by_impact:
                by_impact[impact].append(f)
            else:
                by_impact["informational"].append(f)

        for impact in ["high", "medium", "low", "informational"]:
            if by_impact[impact]:
                lines.append(f"\n### {impact.upper()} Impact:")
                for finding in by_impact[impact]:
                    lines.append(finding.to_prompt_text())

        return "\n".join(lines)

    def get_functions_with_findings(self) -> Dict[str, List[SlitherFinding]]:
        """Extract functions mentioned in findings (GPTScan-style filtering).

        Returns:
            Dict mapping function_name -> list of findings affecting that function
        """
        func_findings: Dict[str, List[SlitherFinding]] = {}

        for finding in self.findings:
            for elem in finding.elements:
                if elem.get("type") == "function":
                    func_name = elem.get("name", "unknown")
                    if func_name not in func_findings:
                        func_findings[func_name] = []
                    if finding not in func_findings[func_name]:
                        func_findings[func_name].append(finding)

        return func_findings

    def get_candidate_functions(self) -> Set[str]:
        """Get set of function names that have potential vulnerabilities.

        Per GPTScan: these are candidate functions for GPT analysis.

        Returns:
            Set of function names with Slither findings
        """
        return set(self.get_functions_with_findings().keys())

    def filter_by_functions(self, function_names: Set[str]) -> 'SlitherResult':
        """Return new SlitherResult with only findings for specified functions.

        Args:
            function_names: Set of function names to include

        Returns:
            New SlitherResult with filtered findings
        """
        filtered_findings = []

        for finding in self.findings:
            has_target_func = False
            for elem in finding.elements:
                if elem.get("type") == "function" and elem.get("name") in function_names:
                    has_target_func = True
                    break

            if has_target_func:
                filtered_findings.append(finding)

        return SlitherResult(
            success=self.success,
            findings=filtered_findings,
            detector_count=len(filtered_findings),
            raw_output=self.raw_output,
        )

    def to_gptscan_prompt_text(self) -> str:
        """Format findings in GPTScan style - grouped by candidate function.

        Per GPTScan paper: static analysis identifies candidate functions,
        which are then passed to GPT for scenario/property matching.

        Returns:
            Formatted string with function-grouped findings
        """
        if not self.success:
            return f"Static analysis failed: {self.error}"

        if not self.findings:
            return "No candidate functions identified by static analysis."

        func_findings = self.get_functions_with_findings()

        if not func_findings:
            return "No function-level findings from static analysis."

        lines = [f"Static analysis identified {len(func_findings)} candidate functions:\n"]

        for func_name, findings in func_findings.items():
            lines.append(f"\n### Function: `{func_name}`")
            lines.append(f"Potential issues ({len(findings)}):")

            for f in findings:
                line_range = f.get_line_range()
                line_str = f"Lines {line_range[0]}-{line_range[1]}" if line_range else "Unknown lines"
                lines.append(f"  - [{f.impact.upper()}] {f.check}: {line_str}")

        return "\n".join(lines)


class SlitherRunner:
    """Runs Slither static analysis on Solidity contracts."""

    def __init__(
        self,
        solc_version: str = "0.8.0",
        timeout: int = 120,
        detectors: Optional[List[str]] = None,
    ):
        """
        Initialize Slither runner.

        Args:
            solc_version: Solidity compiler version to use
            timeout: Timeout in seconds for Slither execution
            detectors: List of specific detectors to run (None = all)
        """
        self.solc_version = solc_version
        self.timeout = timeout
        self.detectors = detectors

    def _detect_solidity_version(self, source: str) -> str:
        """Extract Solidity version from pragma statement."""
        import re

        # Match pragma solidity ^0.8.0; or pragma solidity >=0.7.0 <0.9.0; etc.
        patterns = [
            r'pragma\s+solidity\s*\^?\s*(\d+\.\d+\.\d+)',
            r'pragma\s+solidity\s*>=?\s*(\d+\.\d+\.\d+)',
            r'pragma\s+solidity\s*(\d+\.\d+\.\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return match.group(1)

        return self.solc_version

    def _ensure_solc_version(self, version: str) -> bool:
        """Ensure the required solc version is available."""
        try:
            # Try using solc-select if available
            result = subprocess.run(
                ["solc-select", "use", version, "--always-install"],
                capture_output=True,
                timeout=60
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # solc-select not available, hope the right version is installed
            return True

    def run(self, source: str, filename: str = "contract.sol") -> SlitherResult:
        """
        Run Slither on a Solidity source code string.

        Args:
            source: Solidity source code
            filename: Name to use for the temporary file

        Returns:
            SlitherResult with findings
        """
        # Detect and set Solidity version
        version = self._detect_solidity_version(source)
        self._ensure_solc_version(version)

        # Create temporary file for the contract
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = Path(tmpdir) / filename
            contract_path.write_text(source)

            # Build Slither command
            slither_cmd = _get_slither_executable()
            cmd = [
                slither_cmd,
                str(contract_path),
                "--json", "-",  # Output JSON to stdout
            ]

            # Add specific detectors if configured
            if self.detectors:
                cmd.extend(["--detect", ",".join(self.detectors)])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env={**os.environ, "SOLC_VERSION": version}
                )

                # Parse JSON output
                try:
                    # Slither outputs JSON even on non-zero exit (findings found)
                    output = json.loads(result.stdout) if result.stdout else {}
                except json.JSONDecodeError:
                    # Try stderr if stdout failed
                    try:
                        output = json.loads(result.stderr) if result.stderr else {}
                    except json.JSONDecodeError:
                        return SlitherResult(
                            success=False,
                            findings=[],
                            detector_count=0,
                            error=f"Failed to parse Slither output: {result.stderr[:500]}",
                            raw_output=result.stdout or result.stderr
                        )

                # Extract findings from results
                findings = []
                detectors = output.get("results", {}).get("detectors", [])

                for det in detectors:
                    finding = SlitherFinding(
                        check=det.get("check", "unknown"),
                        impact=det.get("impact", "unknown"),
                        confidence=det.get("confidence", "unknown"),
                        description=det.get("description", ""),
                        elements=det.get("elements", [])
                    )
                    findings.append(finding)

                return SlitherResult(
                    success=True,
                    findings=findings,
                    detector_count=len(detectors),
                    raw_output=result.stdout,
                    _source=source,  # Store source for lazy Slither instance creation
                )

            except subprocess.TimeoutExpired:
                return SlitherResult(
                    success=False,
                    findings=[],
                    detector_count=0,
                    error=f"Slither timed out after {self.timeout} seconds"
                )
            except FileNotFoundError:
                return SlitherResult(
                    success=False,
                    findings=[],
                    detector_count=0,
                    error="Slither not found. Install with: pip install slither-analyzer"
                )
            except Exception as e:
                return SlitherResult(
                    success=False,
                    findings=[],
                    detector_count=0,
                    error=str(e)
                )

    def run_for_prompt(self, source: str) -> str:
        """
        Run Slither and return formatted text for prompt injection.

        Args:
            source: Solidity source code

        Returns:
            Formatted string suitable for P6 prompt template
        """
        result = self.run(source)
        return result.to_prompt_text()


def check_slither_available() -> Tuple[bool, str]:
    """Check if Slither is installed and available."""
    slither_cmd = _get_slither_executable()
    try:
        result = subprocess.run(
            [slither_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Slither {version} is available"
        else:
            return False, "Slither returned error"
    except FileNotFoundError:
        return False, "Slither not found. Install with: pip install slither-analyzer"
    except subprocess.TimeoutExpired:
        return False, "Slither version check timed out"
    except Exception as e:
        return False, f"Error checking Slither: {e}"
