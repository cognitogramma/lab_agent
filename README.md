# Continuum Spectranalytics

Continuum is a research tool for our lab at Tulane. We use it to keep track of
papers and research ideas and to process light-scattering data from the ARGEN.
Each run records the settings and decisions used to produce the results.

### Download for Windows

**[Download Continuum Spectranalytics v0.2.0](https://github.com/cognitogramma/lab_agent/releases/download/v0.2.0/Continuum-Spectranalytics.exe)**

Save the file on a 64-bit Windows PC and double-click it in File Explorer.
Python is bundled. The executable is unsigned. If a Tulane-managed computer
blocks it, contact Tulane IT.

[Release notes and checksum](https://github.com/cognitogramma/lab_agent/releases/tag/v0.2.0)

### Using the app

- **Papers.** Open **Literature**, enter a topic, and select **Find related papers**.
  Choose **Read page** to inspect a result and **Save paper** to keep it.
- **Ideas.** In **Ideas & notes**, select **New research idea** to jot down a
  hypothesis, link the papers that prompted it, and record what to try next.
- **ARGEN data.** Press **Ctrl+O** to open a raw instrument file. The run guide
  explains what the lab needs to supply before calculating molar mass or the
  second virial coefficient.
  [Run guide](docs/reference.md#opening-a-run-and-what-you-have-to-declare)

### Research assistant

Literature search and notes work without a model key. For the assistant, open
**Research assistant > Model settings**, choose Gemini, Claude, or OpenAI,
enter your API key, and select **Use this key**. Your provider may charge for
usage.

Try asking it to compare two papers on light scattering and cite the methods
sections it used. You can also save a specialist to repeat a task across
sessions. [Creating a specialist](docs/reference.md#specialized-subagents)

### Lab and source documentation

- [Working from a source checkout](docs/LAB-SETUP.md)
- [Building the Windows executable](docs/reference.md#build-the-windows-app)
- [Technical reference and known limitations](docs/reference.md)

Tulane University · Advised by Prof. Wayne F. Reed
