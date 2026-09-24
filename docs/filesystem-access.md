# Agent access to local files

Both the GUI and terminal agent expose five tools: `get_file_permissions`,
`list_directory`, `read_file`, `create_directory`, and `write_file`.
In the GUI, **Session > File permissions...** shows the effective policy
without making a model request.

Copy `filesystem-permissions.example.json` to `filesystem-permissions.json`
in the project root, set `enabled` to `true`, and enter absolute read/write
roots. This local file is ignored by Git. Missing, disabled or malformed
configuration refuses file operations. The file tools reload it on each call.

For access across a local Windows drive:

```json
{
  "enabled": true,
  "read_roots": ["C:/"],
  "write_roots": ["C:/"],
  "denied_paths": []
}
```

Use `denied_paths` to exclude additional folders or files; exclusions take
precedence. Add other local drives explicitly if required. Windows account
permissions still apply: this does not grant administrator access.

Built-in exclusions cover Windows, Program Files, ProgramData, AppData,
recovery/system-volume folders, virtual environments, Git internals, common
credential folders, `.env` files, and the permission file itself. Relocated
Windows/application directories from the environment are also excluded.
Network paths, device paths, alternate data streams, symbolic links, junctions,
other reparse points, and hard-linked files are refused. Some OneDrive files
use reparse points; copy those to an ordinary local folder before use.

Examples to type into the GUI (substitute your own paths):

- “Show my file permissions.”
- “List the files in C:\\Users\\YourName\\Documents.”
- “Read C:\\Users\\YourName\\Documents\\experiment-notes.txt.”
- “Save a summary to C:\\Users\\YourName\\Documents\\summary.md.”

Writes and directory creation use the existing commit approval dialog. Existing
`--auto-approve` and session-level “always allow” behavior still applies. A
replacement requires the hash returned by `read_file`, rejects stale versions,
and saves the original beside the file as `.polyrmc-backup-<id>`. Writes are
staged before publication. There are no delete or execute tools.

Text uses UTF-8; binary data can be read/written as base64. Reads return 32 KiB
of characters per page and a next offset; files over 16 MiB are refused. A
single write accepts up to 1 MiB. Scientific data readers remain the intended
way to open larger measurement files. Directories return 100 entries per page.

Read contents enter the model provider's context and the local session log.
The agent should read only files requested or needed for the current task.

This is a boundary for these five tools, **not an OS sandbox for the entire
Python application**. Existing scientific tools retain their own validation;
other software running as your Windows user is not constrained by this file.
Concurrent changes by another process between a path check and disk I/O are
not isolated by this policy. Use OS isolation if that threat matters.

Restart PolyRMC after installing the feature to load the new tool definitions.
Subsequent policy-file changes take effect on the next file operation.
