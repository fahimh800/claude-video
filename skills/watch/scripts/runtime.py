"""Small cross-platform process and stream helpers; no optional dependencies."""
from __future__ import annotations

import os
import subprocess
import sys


def configure_stdio() -> None:
    """Call at script entry only, leaving imported modules' streams untouched."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            try:
                reconfigure(encoding='utf-8', errors='backslashreplace')
            except (OSError, ValueError):
                pass
    refresh_windows_path()


def refresh_windows_path() -> None:
    """Find tools installed since this shell started: winget only updates the registry PATH."""
    if os.name != 'nt':
        return
    try:
        import winreg
    except ImportError:
        return
    entries = []
    for root, key in ((winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'),
                      (winreg.HKEY_CURRENT_USER, 'Environment')):
        try:
            with winreg.OpenKey(root, key) as handle:
                value, _ = winreg.QueryValueEx(handle, 'Path')
        except OSError:
            continue
        entries += [os.path.expandvars(part.strip()) for part in str(value).split(';') if part.strip()]
    if os.environ.get('LOCALAPPDATA'):
        entries.append(os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'WinGet', 'Links'))
    current = os.environ.get('PATH', '')
    known = {part.rstrip('\\/').lower() for part in current.split(os.pathsep) if part}
    extra = [part for part in dict.fromkeys(entries)
             if part.rstrip('\\/').lower() not in known and os.path.isdir(part)]
    if extra:  # Appended, so anything already on PATH keeps priority.
        os.environ['PATH'] = os.pathsep.join([p for p in (current, *extra) if p])


def diagnostic(value, limit: int = 3000) -> str:
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value or '').strip()[-limit:]


def run_text(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Capture diagnostics predictably, including Windows pipes and launch failures."""
    try:
        return subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace', **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f'{cmd[0]} timed out: {diagnostic(exc.stderr)}') from None
    except OSError as exc:
        raise SystemExit(
            f'Cannot run {cmd[0]} ({type(exc).__name__}). Check that the executable is installed '
            'and permitted by your operating system, then retry.'
        ) from None
