# Agent access to local files

## The quick way: Session > File access...

Open **Session > File access...** and tick the folders the assistant may use:

| | read files | save copies here |
| --- | --- | --- |
| Desktop | find and read files there by itself | copy figures, CSVs and attachments there |
| Documents | ... | ... |
| Downloads | ... | ... |

The real Desktop and Documents folders are used, including when OneDrive
syncs them. Saving, copying and opening always ask you first.

You do not need any of this for a single file: drag it onto the window, paste
it with **Ctrl+V** after copying it in File Explorer, or use **Session >
Attach a file...**. An attached file is readable for the rest of the session.

Things to ask once a folder is allowed:

- "What PDFs are on my Desktop?"
- "Open the ARGEN export on my Desktop called run 12 and plot ls8."
- "Copy that figure to my Desktop and open it."
- "Save a summary of this paper to Documents as summary.md."

The tools behind this are `get_file_permissions`, `list_directory`,
`read_file`, `write_file`, `create_directory`, `copy_file` and `open_file`.
`open_file` opens documents, images and folders in their usual application. It
never opens programs or scripts (.exe, .bat, .ps1, .js, macro-enabled Office
files, shortcuts, and similar).

## By hand: filesystem-permissions.json

The dialog writes `filesystem-permissions.json` in the app's data folder
(the project root for a source checkout). You can also write it yourself, set
`enabled` to `true`, and enter absolute read/write roots. Missing, disabled or
malformed configuration refuses file operations. The file tools reload it on
each call.

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
other reparse points, and hard-linked files are refused. OneDrive and other
cloud-synced files are allowed: they are reparse points too, but not links.

Examples to type into the GUI (substitute your own paths):

- “Show my file permissions.”
- “List the files in C:\\Users\\YourName\\Documents.”
- “Read C:\\Users\\YourName\\Documents\\experiment-notes.txt.”
- “Save a summary to C:\\Users\\YourName\\Documents\\summary.md.”

Writes and directory creation use the existing commit approval dialog. Existing
`--auto-approve` and session-level “always allow” behavior still applies. A
replacement requires the hash returned by `read_file`, rejects stale versions,
and saves the original beside the file as `.continuum-spectranalytics-backup-<id>`. Writes are
staged before publication. There are no delete or execute tools.

Text uses UTF-8; binary data can be read/written as base64. Reads return 32 KiB
of characters per page and a next offset; files over 16 MiB are refused. A
single write accepts up to 1 MiB. Scientific data readers remain the intended
way to open larger measurement files. Directories return 100 entries per page.

Read contents enter the model provider's context and the local session log.
The agent should read only files requested or needed for the current task.

This is a boundary for these file tools, **not an OS sandbox for the entire
Python application**. Existing scientific tools retain their own validation;
other software running as your Windows user is not constrained by this file.
Concurrent changes by another process between a path check and disk I/O are
not isolated by this policy. Use OS isolation if that threat matters.

Restart Continuum Spectranalytics after installing the feature to load the new tool definitions.
Subsequent policy-file changes take effect on the next file operation.
