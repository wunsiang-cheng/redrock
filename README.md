# RedRock

RedRock is an autonomous AI agent that operates a computer and communicates through Telegram. It uses DeepSeek `deepseek-v4-pro` by default and preserves state across restarts.

> [!WARNING]
> RedRock can read, change, transmit, and delete files; run arbitrary commands with the installing user's permissions; expose local data and credentials; modify itself; and consume paid services. It has no per-action approval, audit log, rollback, or privacy guarantee. Use a dedicated computer.

Debian and Windows are tested. macOS lifecycle integration is implemented but unverified.

## Install

Download the installer from the [latest release](https://github.com/wunsiang-cheng/redrock/releases/latest).

- **Windows:** Run `redrock-*-windows-x86_64.exe`. Installation is per-user and requires no administrator access.
- **Debian 13:** Install `redrock_*_amd64.deb`, then open **RedRock Setup**.
- **Debian 13 without a desktop:** Install the package, then run `redrock install --cli`.

The graphical installer verifies a DeepSeek API key and Telegram bot token. Send the bot a private message and select **Find me**, or enter Telegram user IDs under **Advanced**.

Both installers store credentials in `config.env`, create `skills/` and `files/inbox/`, configure startup, and start the agent. Default installation directories are:

- Linux: `~/.local/share/redrock`
- macOS: `~/Library/Application Support/RedRock`
- Windows: `%LOCALAPPDATA%\RedRock`

Run a newer installer to update an existing installation while preserving configuration, memory, and allowed users. The graphical installer also provides status, start, stop, reconfiguration, uninstall, and purge controls.

### Headless installation

Run `redrock install --cli`. Without a Linux graphical session, `redrock install` selects the CLI automatically. Enter Telegram user IDs or leave the field empty to detect the next private sender. Credential prompts are hidden.

For non-interactive installation, provide credentials through the environment:

```sh
DEEPSEEK_API_KEY=... \
TELEGRAM_BOT_TOKEN=... \
REDROCK_ALLOWED_USERS=123456789 \
redrock install --cli --non-interactive --accept-risk
```

Use `--directory <path>` to change the installation directory. The CLI requires an active systemd user manager. To keep the service running after logout, enable lingering with `sudo loginctl enable-linger "$USER"`.

## Telegram commands

| Command | Effect |
| --- | --- |
| `/status` | Show version, activity, goal, next wake time, and conversation size. |
| `/reset` | Delete the sender's conversation and stop its active turn. |
| `/nuke confirm` | Delete all conversations, the current goal, and long-term memory. |

Only private messages from IDs in `REDROCK_ALLOWED_USERS` reach the agent. The process reloads this setting on every poll. Commands remain available while a model call or tool is running.

Allowed users can send documents with an optional caption. RedRock stores them under `files/inbox/<user-id>/` and gives the local path to the agent. Telegram limits bot downloads to 20 MB. The agent can send local files up to 50 MB to allowed users.

## Runtime

RedRock runs one model call or tool at a time. User turns suspend autonomous work until the reply finishes. A shell command can block progress for up to one hour.

Available tools are DuckDuckGo search, the platform shell, Telegram messages and files to allowed users, and internal goal, memory, and wake-time updates. Shell output is limited to 64 KiB.

State is saved to SQLite after each step. Interrupted steps use at-least-once execution and may run again after restart. A state-file lock prevents multiple agents from using the same database. Context is compacted at the configured threshold.

The runtime exposes the `skills/` path to the model. Skill files have no required format and are not loaded automatically.

## Build and run

Requires Rust, Cargo, a DeepSeek API key, and a Telegram bot token for Telegram mode.

```sh
cargo build --release
cargo test
packaging/build-deb.sh
```

```sh
DEEPSEEK_API_KEY=... target/release/redrock run "inspect the machine"

DEEPSEEK_API_KEY=... \
TELEGRAM_BOT_TOKEN=... \
REDROCK_ALLOWED_USERS=123456789 \
target/release/redrock telegram
```

## Configuration

Environment variables override values in `config.env` beside the executable.

| Variable | Purpose | Default |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | Model API credential | Required |
| `TELEGRAM_BOT_TOKEN` | Telegram bot credential | Required in Telegram mode |
| `REDROCK_ALLOWED_USERS` | Comma-separated allowed Telegram user IDs | Required in Telegram mode |
| `REDROCK_MODEL` | Model name | `deepseek-v4-pro` |
| `REDROCK_STATE` | SQLite state path | `redrock.db` |
| `REDROCK_SKILLS` | Skill directory | Set by the installer |
| `REDROCK_FILES` | Received-file directory | `files` beside the state database |
| `REDROCK_CONTEXT_RATIO` | Context compaction threshold | `0.5` |
| `REDROCK_CONTEXT_TOKENS` | Model context window | `1000000` |
| `REDROCK_API_BASE` | Responses API endpoint | `https://api.deepseek.com` |
| `TELEGRAM_API_BASE` | Telegram API endpoint | `https://api.telegram.org` |

Providers with a compatible Responses API can be selected with `REDROCK_API_BASE` and `REDROCK_MODEL`. Set `REDROCK_CONTEXT_TOKENS` to the selected model's context window.

## Platform support

| Platform | Lifecycle integration | Status |
| --- | --- | --- |
| Debian Linux | `systemd --user` | Tested |
| macOS | LaunchAgent | Unverified |
| Windows | Per-user `Run` registry entry | Tested; contact detection unverified |

## Constraints

- Allowed users are contacts. Sender IDs come from Telegram or manual configuration.
- Host users can stop the process, revoke credentials, modify files, or remove the installation.
- Each copied installation has independent state.
- `/reset` and `/nuke` do not delete received files. Purge deletes them with the installation directory.
- The installed agent does not include its source code or a self-update API.

## License

[MIT](https://github.com/wunsiang-cheng/redrock/blob/main/LICENSE)
