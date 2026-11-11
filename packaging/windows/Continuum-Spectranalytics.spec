from pathlib import Path
import sysconfig
import tomllib

from PyInstaller.utils.hooks import collect_submodules, copy_metadata
from PyInstaller.utils.win32 import versioninfo


root = Path(SPECPATH).resolve().parents[1]
package = root / "src" / "continuum_spectranalytics"
project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
name = "Continuum Spectranalytics"
version = project["version"]
version_parts = tuple(int(part) for part in version.split(".")) + (0,)
icon = package / "assets" / "continuum.ico"

version_resource = versioninfo.VSVersionInfo(
    ffi=versioninfo.FixedFileInfo(filevers=version_parts, prodvers=version_parts, fileType=1),
    kids=[
        versioninfo.StringFileInfo([
            versioninfo.StringTable("040904B0", [
                versioninfo.StringStruct("FileDescription", name),
                versioninfo.StringStruct("ProductName", name),
                versioninfo.StringStruct("FileVersion", version),
                versioninfo.StringStruct("ProductVersion", version),
                versioninfo.StringStruct("OriginalFilename", f"{name}.exe"),
            ]),
        ]),
        versioninfo.VarFileInfo([versioninfo.VarStruct("Translation", [1033, 1200])]),
    ],
)

# Bundle only read-only application resources. Session data and settings live in AppData.
datas = [
    (str(root / "LAB.md"), "."),
    (str(root / "examples" / "run_config.example.json"), "examples"),
    (str(package / "agent" / "prompts" / "*.md"), "continuum_spectranalytics/agent/prompts"),
    (str(icon), "continuum_spectranalytics/assets"),
]
# The Codex Python runtime loads its certificate distrust list beside ssl.py.
rejected_cas = Path(sysconfig.get_path("stdlib")) / "codex-rejected-cas.pem"
if rejected_cas.is_file():
    datas.append((str(rejected_cas), "."))

for distribution in ("langchain-core", "langchain-openai", "langchain-anthropic",
                     "langchain-google-genai", "openai", "anthropic", "google-genai", "urllib3"):
    datas += copy_metadata(distribution)

# LangChain and Google resolve some public objects on demand. tiktoken discovers plugins.
hidden_imports = collect_submodules("langchain_core") + collect_submodules("google.genai")
hidden_imports.append("tiktoken_ext.openai_public")

analysis = Analysis(
    [str(Path(SPECPATH) / "launch.py")],
    pathex=[str(root / "src")],
    datas=datas,
    hiddenimports=hidden_imports,
    hooksconfig={"matplotlib": {"backends": ["Agg", "TkAgg"]}},
)
archive = PYZ(analysis.pure)
exe = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    name=name,
    console=False,
    upx=False,
    icon=str(icon),
    version=version_resource,
)
