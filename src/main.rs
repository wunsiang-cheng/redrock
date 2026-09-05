#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod config;
mod installer_cli;
mod state;

use config::*;
use installer_cli::run_install_command;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use state::*;
use std::{
    cell::Cell,
    collections::HashSet,
    env,
    error::Error,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DUCKDUCKGO_LITE: &str = "https://lite.duckduckgo.com/lite/";
const MAX_TIMEOUT: u64 = 3600;
const MAX_OUTPUT: usize = 64 * 1024;
const MAX_RESPONSE_TOKENS: usize = 32_768;
const MIN_WAKE_SECONDS: u64 = 30;
const MAX_MESSAGE_CHARS: usize = 4096;
const MAX_CAPTION_CHARS: usize = 1024;
const MAX_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;
const MAX_UPLOAD_BYTES: u64 = 50 * 1024 * 1024;
const POLL_RETRY_SECONDS: u64 = 5;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArgs {
    command: String,
    working_directory: String,
    timeout_seconds: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageArgs {
    user_id: i64,
    text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileArgs {
    user_id: i64,
    path: String,
    caption: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuityArgs {
    current_goal: String,
    long_term_memory: String,
    wake_in_seconds: u64,
}

struct Telegram {
    client: Client,
    token: String,
    api: String,
    allowed: HashSet<i64>,
    /// The user who started this turn.
    watcher: Option<i64>,
    /// The posted progress message.
    progress: Cell<Option<i64>>,
    /// Whether the agent has replied.
    spoke: Cell<bool>,
    started: i64,
    steps: u32,
    /// Whether the turn was cancelled while its step was running.
    dropped: Arc<AtomicBool>,
}

impl Telegram {
    fn dropped(&self) -> bool {
        self.dropped.load(Ordering::Relaxed)
    }
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("run") => {
            let goal = args.collect::<Vec<_>>().join(" ");
            if goal.is_empty() {
                return Err("goal must not be empty".into());
            }
            let key = required_env("DEEPSEEK_API_KEY")?;
            let api = configured("REDROCK_API_BASE").unwrap_or_else(|| DEEPSEEK_API.into());
            let mut history = vec![json!({"role":"user", "content":goal})];
            let mut continuity = Continuity::default();
            println!(
                "{}",
                agent_turn(
                    &client()?,
                    &mut history,
                    &mut continuity,
                    &key,
                    &api,
                    None,
                    context_budget()?,
                )?
            );
        }
        Some("telegram") => {
            if args.next().is_some() {
                return Err("usage: redrock telegram".into());
            }
            run_telegram()?;
        }
        Some("install") => run_install_command(args.collect())?,
        None => run_install_command(Vec::new())?,
        _ => {
            return Err(
                "usage: redrock run \"<goal>\" | redrock telegram | redrock install [--cli | --gui]".into(),
            );
        }
    }
    Ok(())
}

const DISCLOSURE: &str = "RedRock can read, alter, transmit, or delete files available to your account; execute arbitrary commands and attempt to use existing sudo access; expose local content, credentials, command results, and conversations to DeepSeek or contacts; modify itself; and consume compute, storage, network, and paid API quota. It acts unpredictably without per-action approval, audit history, recovery, or privacy guarantees.";

/// Installer screen.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Disclosure,
    Key,
    Token,
    Contact,
    Confirm,
    Console,
}

impl Stage {
    fn next(self) -> Self {
        match self {
            Self::Disclosure => Self::Key,
            Self::Key => Self::Token,
            Self::Token => Self::Contact,
            Self::Contact => Self::Confirm,
            Self::Confirm | Self::Console => Self::Console,
        }
    }

    fn back(self) -> Option<Self> {
        match self {
            Self::Key => Some(Self::Disclosure),
            Self::Token => Some(Self::Key),
            Self::Contact => Some(Self::Token),
            Self::Confirm => Some(Self::Contact),
            _ => None,
        }
    }
}

/// Result of an installer step.
struct Step {
    message: String,
    users: Option<String>,
}

impl From<String> for Step {
    fn from(message: String) -> Self {
        Self {
            message,
            users: None,
        }
    }
}

struct Installer {
    stage: Stage,
    directory: String,
    deepseek_key: String,
    telegram_token: String,
    telegram_users: String,
    acknowledged: bool,
    status: String,
    task: Option<(&'static str, Receiver<std::result::Result<Step, String>>)>,
}

impl Default for Installer {
    fn default() -> Self {
        let directory = default_install_directory().to_string_lossy().into_owned();
        Self {
            stage: if is_installation(Path::new(&directory)) {
                Stage::Console
            } else {
                Stage::Disclosure
            },
            deepseek_key: String::new(),
            telegram_token: String::new(),
            telegram_users: config_value(Path::new(&directory), "REDROCK_ALLOWED_USERS")
                .unwrap_or_default(),
            directory,
            acknowledged: false,
            status: String::new(),
            task: None,
        }
    }
}

impl Installer {
    /// Run a network operation outside the UI thread.
    fn start(
        &mut self,
        message: &'static str,
        work: impl FnOnce() -> Result<Step> + Send + 'static,
    ) {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let _ = sender.send(work().map_err(|error| error.to_string()));
        });
        self.status.clear();
        self.task = Some((message, receiver));
    }

    /// Render the active step. Returns whether it owns the screen.
    fn working(&mut self, ui: &mut eframe::egui::Ui) -> bool {
        let progress = self
            .task
            .as_ref()
            .map(|(message, task)| (*message, task.try_recv()));
        match progress {
            Some((message, Err(mpsc::TryRecvError::Empty))) => {
                ui.horizontal(|ui| {
                    ui.add(eframe::egui::Spinner::new());
                    ui.label(message);
                });
                ui.ctx().request_repaint();
                return true;
            }
            Some((_, Ok(result))) => {
                self.task = None;
                match result {
                    Ok(step) => {
                        if let Some(users) = step.users {
                            self.telegram_users = users;
                        }
                        self.status = step.message;
                        self.stage = self.stage.next();
                    }
                    Err(error) => self.status = format!("Error: {error}"),
                }
            }
            Some((_, Err(mpsc::TryRecvError::Disconnected))) => {
                self.task = None;
                self.status = "Error: the step ended without a result.".into();
            }
            None => {}
        }
        false
    }

    /// Render navigation. Returns whether the user selected Next.
    fn navigation(&mut self, ui: &mut eframe::egui::Ui, label: &str, ready: bool) -> bool {
        let back = self.stage.back();
        let mut forward = false;
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if let Some(back) = back
                && ui.button("Back").clicked()
            {
                self.stage = back;
                self.status.clear();
            }
            forward = ui
                .add_enabled(ready, eframe::egui::Button::new(label))
                .clicked();
        });
        forward
    }

    fn secret(ui: &mut eframe::egui::Ui, value: &mut String, hint: &str) {
        ui.add(
            eframe::egui::TextEdit::singleline(value)
                .password(true)
                .hint_text(hint)
                .desired_width(400.0),
        );
    }
}

impl eframe::App for Installer {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _: &mut eframe::Frame) {
        eframe::egui::Frame::new()
            .inner_margin(20)
            .show(ui, |ui| self.screen(ui));
    }
}

impl Installer {
    /// Render the current installer stage.
    fn screen(&mut self, ui: &mut eframe::egui::Ui) {
        ui.heading("RedRock");
        ui.add_space(12.0);
        if self.working(ui) {
            return;
        }

        match self.stage {
            Stage::Disclosure => self.disclosure(ui),
            Stage::Key => self.key(ui),
            Stage::Token => self.token(ui),
            Stage::Contact => self.contact(ui),
            Stage::Confirm => self.confirm(ui),
            Stage::Console => self.console(ui),
        }

        if !self.status.is_empty() {
            ui.add_space(12.0);
            ui.separator();
            ui.label(&self.status);
        }
    }

    fn disclosure(&mut self, ui: &mut eframe::egui::Ui) {
        ui.strong("Before you install");
        ui.add_space(4.0);
        ui.label(DISCLOSURE);
        ui.add_space(8.0);
        ui.checkbox(
            &mut self.acknowledged,
            "I understand and authorize ongoing operation.",
        );
        ui.add_space(8.0);
        if ui
            .add_enabled(self.acknowledged, eframe::egui::Button::new("Continue"))
            .clicked()
        {
            self.stage = self.stage.next();
        }
    }

    fn key(&mut self, ui: &mut eframe::egui::Ui) {
        ui.strong("Step 1 of 4 · DeepSeek API key");
        ui.label("Create one at platform.deepseek.com, under API keys.");
        ui.add_space(8.0);
        Self::secret(ui, &mut self.deepseek_key, "sk-…");
        let ready = !self.deepseek_key.is_empty();
        if self.navigation(ui, "Next", ready) {
            let key = self.deepseek_key.clone();
            self.start("Checking the DeepSeek key…", move || {
                verify_deepseek(&key).map(Step::from)
            });
        }
    }

    fn token(&mut self, ui: &mut eframe::egui::Ui) {
        ui.strong("Step 2 of 4 · Telegram bot token");
        ui.label("Message @BotFather on Telegram and send /newbot.");
        ui.add_space(8.0);
        Self::secret(ui, &mut self.telegram_token, "123456789:AA…");
        let ready = !self.telegram_token.is_empty();
        if self.navigation(ui, "Next", ready) {
            let token = self.telegram_token.clone();
            self.start("Checking the bot token…", move || {
                verify_telegram(&token).map(Step::from)
            });
        }
    }

    fn contact(&mut self, ui: &mut eframe::egui::Ui) {
        ui.strong("Step 3 of 4 · Who may contact RedRock");
        ui.label(
            "RedRock ignores every Telegram user except the ones listed here. Open a \
             private chat with your bot, send it any message, then press Find me.",
        );
        ui.add_space(8.0);
        if self.telegram_users.is_empty() {
            ui.label("No Telegram user IDs listed yet.");
        } else {
            ui.label("Allowed Telegram user IDs:");
            ui.add(
                eframe::egui::Label::new(
                    eframe::egui::RichText::new(&self.telegram_users)
                        .monospace()
                        .size(20.0),
                )
                .selectable(true),
            );
        }
        ui.add_space(8.0);
        if ui.button("Find me").clicked() {
            let (token, listed) = (self.telegram_token.clone(), self.telegram_users.clone());
            self.start("Waiting for a message to your bot…", move || {
                capture_contact(&token, &listed)
            });
        }
        ui.add_space(8.0);
        eframe::egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
            ui.label("Allowed Telegram user IDs, separated by commas");
            ui.add(
                eframe::egui::TextEdit::singleline(&mut self.telegram_users).desired_width(400.0),
            );
        });
        let ready = !self.telegram_users.is_empty();
        if self.navigation(ui, "Next", ready) {
            self.stage = self.stage.next();
            self.status.clear();
        }
    }

    fn confirm(&mut self, ui: &mut eframe::egui::Ui) {
        ui.strong("Step 4 of 4 · Install");
        ui.label("RedRock will answer these Telegram user IDs, and no others:");
        ui.add(
            eframe::egui::Label::new(eframe::egui::RichText::new(&self.telegram_users).monospace())
                .selectable(true),
        );
        ui.add_space(4.0);
        ui.label("It will be installed for your user account only:");
        ui.add(
            eframe::egui::Label::new(eframe::egui::RichText::new(&self.directory).monospace())
                .selectable(true),
        );
        ui.add_space(8.0);
        eframe::egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
            ui.label("Installation directory");
            ui.add(eframe::egui::TextEdit::singleline(&mut self.directory).desired_width(400.0));
        });
        let ready = !self.directory.is_empty() && !self.telegram_users.is_empty();
        if self.navigation(ui, "Install", ready) {
            let (directory, key, token, users) = (
                self.directory.clone(),
                self.deepseek_key.clone(),
                self.telegram_token.clone(),
                self.telegram_users.clone(),
            );
            self.start("Installing and starting RedRock…", move || {
                install(&directory, &key, &token, &users)
                    .map(|()| "Installed and started.".to_owned().into())
            });
        }
    }

    /// Render controls for an existing installation.
    fn console(&mut self, ui: &mut eframe::egui::Ui) {
        ui.strong("RedRock is installed");
        ui.add(
            eframe::egui::Label::new(eframe::egui::RichText::new(&self.directory).monospace())
                .selectable(true),
        );
        let carried = env!("CARGO_PKG_VERSION");
        let installed = installed_version(Path::new(&self.directory));
        let outdated = installed.as_deref() != Some(carried);
        ui.label(match &installed {
            Some(version) if !outdated => format!("Version {version}"),
            Some(version) => format!("Version {version} · this installer carries {carried}"),
            None => format!("This installer carries {carried}"),
        });
        ui.add_space(8.0);
        let mut action = None;
        let mut update_now = false;
        ui.horizontal(|ui| {
            for (label, name) in [("Status", "status"), ("Start", "start"), ("Stop", "stop")] {
                if ui.button(label).clicked() {
                    action = Some(name);
                }
            }
            if outdated {
                update_now = ui.button(format!("Update to {carried}")).clicked();
            }
        });
        ui.add_space(8.0);
        let mut reconfigure = false;
        eframe::egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
            ui.label(
                "Uninstall removes the program and keeps your credentials and memory. \
                 Purge deletes the whole installation directory.",
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if !outdated {
                    update_now = ui.button("Reinstall").clicked();
                }
                reconfigure = ui.button("Change credentials").clicked();
                if ui.button("Uninstall").clicked() {
                    action = Some("uninstall");
                }
                if ui.button("Purge").clicked() {
                    action = Some("purge");
                }
            });
        });
        if reconfigure {
            self.stage = Stage::Key;
            self.status.clear();
        }
        if update_now {
            let directory = self.directory.clone();
            self.start("Installing and restarting RedRock…", move || {
                update(&directory).map(Step::from)
            });
        }
        if let Some(name) = action {
            let directory = PathBuf::from(&self.directory);
            self.start("Working…", move || {
                lifecycle(name, &directory).map(Step::from)
            });
        }
    }
}

fn is_installation(directory: &Path) -> bool {
    directory.join(".redrock-install").is_file() || has_installation_layout(directory)
}

fn has_installation_layout(directory: &Path) -> bool {
    directory.join(binary_name()).is_file()
        && directory.join("config.env").is_file()
        && directory.join("skills").is_dir()
        && [
            "DEEPSEEK_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "REDROCK_STATE",
            "REDROCK_SKILLS",
        ]
        .iter()
        .all(|name| config_value(directory, name).is_some())
}

/// Read a semantic version from an installation marker.
fn installed_version(directory: &Path) -> Option<String> {
    let marker = fs::read_to_string(directory.join(".redrock-install")).ok()?;
    let version = marker.trim().to_string();
    (!version.is_empty() && !version.contains(' ')).then_some(version)
}

/// Load credentials from an existing installation.
fn update(directory: &str) -> Result<String> {
    let key = config_value(Path::new(directory), "DEEPSEEK_API_KEY")
        .ok_or("config.env holds no DeepSeek API key")?;
    let token = config_value(Path::new(directory), "TELEGRAM_BOT_TOKEN")
        .ok_or("config.env holds no Telegram bot token")?;
    // Older installations may not contain an allow list.
    let users = config_value(Path::new(directory), "REDROCK_ALLOWED_USERS").ok_or(
        "config.env lists no allowed Telegram user IDs; use Change credentials under Advanced",
    )?;
    install(directory, &key, &token, &users)?;
    Ok(format!("Now running {}.", env!("CARGO_PKG_VERSION")))
}

fn verify_deepseek(key: &str) -> Result<String> {
    let response = client()?
        .get(format!("{DEEPSEEK_API}/models"))
        .bearer_auth(key)
        .send()?;
    if !response.status().is_success() {
        return Err(format!("DeepSeek rejected the API key ({})", response.status()).into());
    }
    Ok(String::new())
}

fn verify_telegram(token: &str) -> Result<String> {
    let bot = telegram_call(&client()?, TELEGRAM_API, token, "getMe", &json!({}))?;
    Ok(match bot["result"]["username"].as_str() {
        Some(username) => format!("Connected to @{username}."),
        None => String::new(),
    })
}

fn run_installer() -> Result<()> {
    eframe::run_native(
        "RedRock Installation",
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default().with_inner_size([560.0, 420.0]),
            ..Default::default()
        },
        Box::new(|_| Ok(Box::new(Installer::default()))),
    )
    .map_err(|error| error.to_string().into())
}

/// Return the sender of the next private Telegram message.
fn capture_contact(token: &str, listed: &str) -> Result<Step> {
    let client = client()?;
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut offset = 0_i64;
    while Instant::now() < deadline {
        let updates = telegram_call(
            &client,
            TELEGRAM_API,
            token,
            "getUpdates",
            &json!({"offset":offset,"timeout":20,"allowed_updates":["message"]}),
        )?;
        for update in updates["result"]
            .as_array()
            .ok_or("Telegram response is missing result")?
        {
            if let Some(id) = update["update_id"].as_i64() {
                offset = offset.max(id + 1);
            }
            let message = &update["message"];
            if message["chat"]["type"] != "private" {
                continue;
            }
            let Some(user_id) = message["from"]["id"].as_i64() else {
                continue;
            };
            let who = message["from"]["username"].as_str().map_or_else(
                || user_id.to_string(),
                |name| format!("@{name} ({user_id})"),
            );
            return Ok(Step {
                message: format!("Found {who}. Install only if that is you."),
                users: Some(with_user(listed, user_id)),
            });
        }
    }
    Err(
        "No message reached the bot within two minutes. Send it one, then press Find me again."
            .into(),
    )
}

/// Add a user ID without removing existing IDs.
fn with_user(listed: &str, user_id: i64) -> String {
    let id = user_id.to_string();
    let mut users: Vec<&str> = listed
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .collect();
    if !users.contains(&id.as_str()) {
        users.push(&id);
    }
    users.join(",")
}

#[cfg(target_os = "linux")]
fn default_install_directory() -> PathBuf {
    PathBuf::from(env::var("HOME").unwrap_or_default()).join(".local/share/redrock")
}

#[cfg(target_os = "macos")]
fn default_install_directory() -> PathBuf {
    PathBuf::from(env::var("HOME").unwrap_or_default()).join("Library/Application Support/RedRock")
}

#[cfg(target_os = "windows")]
fn default_install_directory() -> PathBuf {
    PathBuf::from(env::var("LOCALAPPDATA").unwrap_or_default()).join("RedRock")
}

#[cfg(target_os = "linux")]
fn service_file(_: &Path) -> Result<PathBuf> {
    Ok(PathBuf::from(required_env("HOME")?).join(".config/systemd/user/redrock.service"))
}

#[cfg(target_os = "macos")]
fn service_file(_: &Path) -> Result<PathBuf> {
    Ok(PathBuf::from(required_env("HOME")?).join("Library/LaunchAgents/com.redrock.agent.plist"))
}

#[cfg(target_os = "windows")]
fn service_file(directory: &Path) -> Result<PathBuf> {
    Ok(directory.join("redrock-task.txt"))
}

fn install(directory: &str, key: &str, token: &str, users: &str) -> Result<()> {
    if directory.is_empty() || key.is_empty() || token.is_empty() {
        return Err("all fields are required".into());
    }
    if [key, token, users]
        .iter()
        .any(|value| value.contains(['\n', '\r']))
    {
        return Err("credentials and the allowed-user list must be one line".into());
    }
    // Validate the allow list before installation.
    parse_allowed_users(users)?;
    verify_deepseek(key)?;
    verify_telegram(token)?;

    let directory = Path::new(directory);
    let service = service_file(directory)?;
    platform_prepare_install()?;
    write_install_files(directory, key, token, users, &env::current_exe()?, &service)?;
    platform_install(directory, &service)?;
    Ok(())
}

fn write_install_files(
    directory: &Path,
    key: &str,
    token: &str,
    users: &str,
    source_binary: &Path,
    service: &Path,
) -> Result<()> {
    let marker = directory.join(".redrock-install");
    if directory.exists() && !marker.exists() && directory.read_dir()?.next().is_some() {
        return Err("installation directory is not empty".into());
    }
    fs::create_dir_all(directory)?;
    let binary = directory.join(binary_name());
    let replacement = directory.join(".redrock-new");
    let running_installed_binary =
        binary.exists() && fs::canonicalize(source_binary)? == fs::canonicalize(&binary)?;
    if !running_installed_binary {
        fs::copy(source_binary, &replacement)?;
        private_permissions(&replacement, 0o755)?;
        replace_binary(&replacement, &binary)?;
    }
    fs::write(&marker, format!("{}\n", env!("CARGO_PKG_VERSION")))?;
    let skills = directory.join("skills");
    let files = directory.join("files");
    fs::create_dir_all(&skills)?;
    fs::create_dir_all(files.join("inbox"))?;
    let config = directory.join("config.env");
    fs::write(
        &config,
        format!(
            "DEEPSEEK_API_KEY={}\nTELEGRAM_BOT_TOKEN={}\nREDROCK_ALLOWED_USERS={}\nREDROCK_STATE={}\nREDROCK_SKILLS={}\nREDROCK_FILES={}\n",
            env_value(key),
            env_value(token),
            env_value(users),
            env_value(&directory.join("redrock.db").to_string_lossy()),
            env_value(&skills.to_string_lossy()),
            env_value(&files.to_string_lossy())
        ),
    )?;
    private_permissions(&config, 0o600)?;

    fs::create_dir_all(service.parent().ok_or("invalid service path")?)?;
    fs::write(service, service_definition(directory, &binary, &config))?;
    Ok(())
}

fn env_value(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(not(target_os = "windows"))]
fn binary_name() -> &'static str {
    "redrock"
}

#[cfg(unix)]
fn replace_binary(replacement: &Path, binary: &Path) -> Result<()> {
    fs::rename(replacement, binary)?;
    Ok(())
}

#[cfg(windows)]
fn replace_binary(replacement: &Path, binary: &Path) -> Result<()> {
    // Stop the installed agent before replacing its executable.
    let _ = command_output("taskkill.exe", &["/F", "/IM", binary_name()]);
    // Move the old Windows executable aside before replacement.
    if binary.exists() {
        let previous = binary.with_extension("old");
        let _ = fs::remove_file(&previous);
        fs::rename(binary, &previous)?;
    }
    fs::rename(replacement, binary)?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn binary_name() -> &'static str {
    "redrock.exe"
}

#[cfg(unix)]
fn private_permissions(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(windows)]
fn private_permissions(_: &Path, _: u32) -> Result<()> {
    // Windows uses the user's inherited ACL.
    Ok(())
}

#[cfg(target_os = "linux")]
fn systemd_quote(path: &Path) -> String {
    env_value(&path.to_string_lossy())
}

#[cfg(target_os = "linux")]
fn systemctl(arguments: &[&str]) -> Result<String> {
    let output = Command::new("systemctl")
        .arg("--user")
        .args(arguments)
        .output()?;
    let text = format!("{}{}", bounded(&output.stdout), bounded(&output.stderr));
    if !output.status.success() {
        return Err(format!("systemctl {} failed: {text}", arguments.join(" ")).into());
    }
    Ok(text.trim().to_owned())
}

#[cfg(target_os = "macos")]
fn command_output(program: &str, arguments: &[&str]) -> Result<String> {
    let output = Command::new(program).args(arguments).output()?;
    let text = format!("{}{}", bounded(&output.stdout), bounded(&output.stderr));
    if !output.status.success() {
        return Err(format!("{program} {} failed: {text}", arguments.join(" ")).into());
    }
    Ok(text.trim().to_owned())
}

#[cfg(target_os = "windows")]
fn command_output(program: &str, arguments: &[&str]) -> Result<String> {
    let output = windowless(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed with exit code {}",
            output.status.code().unwrap_or(-1)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(target_os = "linux")]
fn service_definition(directory: &Path, binary: &Path, config: &Path) -> String {
    format!(
        "[Unit]\nDescription=RedRock autonomous agent\nAfter=network-online.target\n\n[Service]\nType=simple\nEnvironmentFile={}\nWorkingDirectory={}\nExecStart={} telegram\nRestart=always\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        config.display(),
        directory.display(),
        systemd_quote(binary),
    )
}

#[cfg(target_os = "linux")]
fn platform_prepare_install() -> Result<()> {
    systemctl(&["show-environment"]).map(|_| ()).map_err(|error| {
        format!(
            "cannot connect to the systemd user manager; log in through a user session and verify XDG_RUNTIME_DIR: {error}"
        )
        .into()
    })
}

#[cfg(target_os = "linux")]
fn platform_install(_: &Path, _: &Path) -> Result<()> {
    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", "redrock.service"])?;
    systemctl(&["restart", "redrock.service"])?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn platform_action(action: &str, _: &Path) -> Result<String> {
    let verb = if action == "status" {
        "is-active"
    } else {
        action
    };
    systemctl(&[verb, "redrock.service"]).map(|output| {
        if output.is_empty() {
            format!("{action}ed.")
        } else {
            output
        }
    })
}

#[cfg(target_os = "linux")]
fn platform_remove() -> Result<()> {
    let _ = systemctl(&["disable", "--now", "redrock.service"]);
    Ok(())
}

#[cfg(target_os = "linux")]
fn platform_refresh() -> Result<()> {
    systemctl(&["daemon-reload"])?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn xml(value: &Path) -> String {
    value
        .to_string_lossy()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(target_os = "macos")]
fn service_definition(directory: &Path, binary: &Path, _: &Path) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict><key>Label</key><string>com.redrock.agent</string><key>ProgramArguments</key><array><string>{}</string><string>telegram</string></array><key>WorkingDirectory</key><string>{}</string><key>KeepAlive</key><true/><key>RunAtLoad</key><true/></dict></plist>\n",
        xml(binary),
        xml(directory)
    )
}

#[cfg(target_os = "macos")]
fn launch_domain() -> Result<String> {
    Ok(format!("gui/{}", command_output("id", &["-u"])?))
}

#[cfg(target_os = "macos")]
fn platform_prepare_install() -> Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn platform_install(_: &Path, service: &Path) -> Result<()> {
    let domain = launch_domain()?;
    let _ = command_output(
        "launchctl",
        &["bootout", &domain, service.to_string_lossy().as_ref()],
    );
    command_output(
        "launchctl",
        &["bootstrap", &domain, service.to_string_lossy().as_ref()],
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn platform_action(action: &str, _: &Path) -> Result<String> {
    let target = format!("{}/com.redrock.agent", launch_domain()?);
    match action {
        "status" => command_output("launchctl", &["print", &target]),
        "start" => command_output("launchctl", &["kickstart", &target]),
        "stop" => command_output("launchctl", &["kill", "SIGTERM", &target]),
        _ => unreachable!(),
    }
}

#[cfg(target_os = "macos")]
fn platform_remove() -> Result<()> {
    let target = format!("{}/com.redrock.agent", launch_domain()?);
    let _ = command_output("launchctl", &["bootout", &target]);
    Ok(())
}

#[cfg(target_os = "macos")]
fn platform_refresh() -> Result<()> {
    Ok(())
}

#[cfg(target_os = "windows")]
fn service_definition(_: &Path, binary: &Path, _: &Path) -> String {
    format!("{}\n", windows_autostart_command(binary))
}

#[cfg(any(target_os = "windows", test))]
fn windows_autostart_command(binary: &Path) -> String {
    format!("\"{}\" telegram", binary.display())
}

#[cfg(target_os = "windows")]
const WINDOWS_RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";

#[cfg(target_os = "windows")]
fn start_windows_agent(binary: &Path) -> Result<()> {
    Command::new(binary)
        .arg("telegram")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(())
}

/// Return whether another Windows agent process is running.
#[cfg(target_os = "windows")]
fn windows_agent_running() -> bool {
    command_output(
        "tasklist.exe",
        &[
            "/NH",
            "/FI",
            &format!("IMAGENAME eq {}", binary_name()),
            "/FI",
            &format!("PID ne {}", std::process::id()),
        ],
    )
    .is_ok_and(|output| output.contains(binary_name()))
}

#[cfg(target_os = "windows")]
fn platform_prepare_install() -> Result<()> {
    let _ = command_output("schtasks.exe", &["/End", "/TN", "RedRock"]);
    let _ = command_output("schtasks.exe", &["/Delete", "/F", "/TN", "RedRock"]);
    Ok(())
}

#[cfg(target_os = "windows")]
fn platform_install(directory: &Path, _: &Path) -> Result<()> {
    let binary = directory.join(binary_name());
    let command = windows_autostart_command(&binary);
    command_output(
        "reg.exe",
        &[
            "add",
            WINDOWS_RUN_KEY,
            "/v",
            "RedRock",
            "/t",
            "REG_SZ",
            "/d",
            &command,
            "/f",
        ],
    )?;
    start_windows_agent(&binary)?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn platform_action(action: &str, directory: &Path) -> Result<String> {
    match action {
        "status" => command_output("reg.exe", &["query", WINDOWS_RUN_KEY, "/v", "RedRock"])
            .map(|_| "Installed and configured to start at sign-in.".into()),
        "start" if windows_agent_running() => Ok("Already running.".into()),
        "start" => {
            start_windows_agent(&directory.join(binary_name()))?;
            Ok("Started.".into())
        }
        "stop" => {
            let _ = command_output("taskkill.exe", &["/F", "/IM", binary_name()]);
            Ok("Stopped.".into())
        }
        _ => unreachable!(),
    }
}

#[cfg(target_os = "windows")]
fn platform_remove() -> Result<()> {
    let _ = command_output("schtasks.exe", &["/End", "/TN", "RedRock"]);
    let _ = command_output("schtasks.exe", &["/Delete", "/F", "/TN", "RedRock"]);
    let _ = command_output(
        "reg.exe",
        &["delete", WINDOWS_RUN_KEY, "/v", "RedRock", "/f"],
    );
    let _ = command_output("taskkill.exe", &["/F", "/IM", binary_name()]);
    Ok(())
}

#[cfg(target_os = "windows")]
fn platform_refresh() -> Result<()> {
    Ok(())
}

fn lifecycle(action: &str, directory: &Path) -> Result<String> {
    match action {
        "status" | "start" | "stop" => platform_action(action, directory),
        "uninstall" | "purge" => {
            if !is_installation(directory) {
                return Err("not a RedRock installation directory".into());
            }
            if action == "uninstall" && !directory.join(".redrock-install").is_file() {
                fs::write(
                    directory.join(".redrock-install"),
                    format!("{}\n", env!("CARGO_PKG_VERSION")),
                )?;
            }
            platform_remove()?;
            let service = service_file(directory)?;
            if service.exists() {
                fs::remove_file(service)?;
            }
            platform_refresh()?;
            if action == "purge" {
                fs::remove_dir_all(directory)?;
                Ok("Purged program, configuration, credentials, and memory.".into())
            } else {
                let binary = directory.join(binary_name());
                let _ = fs::remove_file(binary.with_extension("old"));
                if binary.exists() {
                    fs::remove_file(binary)?;
                }
                Ok("Uninstalled program; configuration and state retained.".into())
            }
        }
        _ => Err("unknown lifecycle action".into()),
    }
}

/// Event received by the state-owning loop.
enum Event {
    /// A Telegram message.
    Update(Value),
    /// The next Telegram update offset.
    Offset(i64),
    /// A completed Telegram document download.
    FileDownloaded(Box<FileDownloadResult>),
    /// A completed agent step.
    Stepped(Box<StepResult>),
}

#[derive(Debug)]
struct ReceivedFile {
    path: PathBuf,
    size: u64,
}

struct FileDownloadResult {
    update_id: i64,
    result: std::result::Result<ReceivedFile, String>,
}

/// A running agent step.
struct Running {
    user: Option<i64>,
    read_to: usize,
    dropped: Arc<AtomicBool>,
}

impl Running {
    fn new(request: &StepRequest, read_to: usize) -> Self {
        Self {
            user: request.telegram.watcher,
            read_to,
            dropped: request.telegram.dropped.clone(),
        }
    }

    /// Whether the step's turn was cancelled.
    fn is_stale(&self, state: &State) -> bool {
        match self.user {
            Some(user) => !state
                .active
                .as_ref()
                .is_some_and(|active| active.user == user),
            None => !state.waking,
        }
    }
}

/// Long-poll Telegram and send updates to the main loop.
fn spawn_poller(agent: &Agent, offset: i64, events: mpsc::Sender<Event>) {
    let (client, api, token) = (
        agent.client.clone(),
        agent.telegram_api.clone(),
        agent.token.clone(),
    );
    thread::spawn(move || {
        let mut offset = offset;
        loop {
            let body = json!({"offset":offset,"timeout":30,"allowed_updates":["message"]});
            // Retry transient Telegram failures.
            let response = match telegram_call(&client, &api, &token, "getUpdates", &body) {
                Ok(response) => response,
                Err(error) => {
                    eprintln!("polling Telegram failed: {error}");
                    thread::sleep(Duration::from_secs(POLL_RETRY_SECONDS));
                    continue;
                }
            };
            let Some(updates) = response["result"].as_array() else {
                eprintln!("Telegram getUpdates returned no result array");
                thread::sleep(Duration::from_secs(POLL_RETRY_SECONDS));
                continue;
            };
            for update in updates {
                if let Some(id) = update["update_id"].as_i64() {
                    offset = offset.max(id + 1);
                }
                // Stop when the main loop closes the channel.
                if events.send(Event::Update(update.clone())).is_err() {
                    return;
                }
            }
            if !updates.is_empty() && events.send(Event::Offset(offset)).is_err() {
                return;
            }
        }
    });
}

fn run_telegram() -> Result<()> {
    let path = configured("REDROCK_STATE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("redrock.db"));
    let files = configured("REDROCK_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join("files")
        });
    let agent = Agent {
        client: client()?,
        key: required_env("DEEPSEEK_API_KEY")?,
        token: required_env("TELEGRAM_BOT_TOKEN")?,
        model_api: configured("REDROCK_API_BASE").unwrap_or_else(|| DEEPSEEK_API.into()),
        telegram_api: configured("TELEGRAM_API_BASE").unwrap_or_else(|| TELEGRAM_API.into()),
        budget: context_budget()?,
        files,
    };
    let database = open_database(&path)?;
    let mut state = load_state(&database)?;
    let mut allowed = allowed_users()?;
    // Delay the first autonomous turn for a new state.
    if state.continuity.next_wake == 0 {
        state.continuity.next_wake = now() + MIN_WAKE_SECONDS as i64;
        save_state(&database, &state)?;
    }
    // Populate the Telegram command menu when available.
    let _ = telegram_call(
        &agent.client,
        &agent.telegram_api,
        &agent.token,
        "setMyCommands",
        &json!({"commands":[
            {"command":"status","description":"Show version, goal, and next wake-up"},
            {"command":"reset","description":"Clear this conversation"},
            {"command":"nuke","description":"Erase all conversations, goal, and memory"}
        ]}),
    );
    let (events, arriving) = mpsc::channel();
    spawn_poller(&agent, state.telegram_offset, events.clone());
    // Only one step runs at a time.
    let mut running: Option<Running> = None;
    let mut downloading: Option<i64> = None;
    loop {
        // Reload a valid allow list without restarting.
        allowed = allowed_users().unwrap_or(allowed);
        if downloading.is_none()
            && let Some(file) = state.pending_files.first().cloned()
        {
            downloading = Some(file.update_id);
            spawn_file_download(&agent, file, events.clone());
        }
        if running.is_none() {
            begin_turn(&agent, &mut state);
            // User turns take priority over autonomous work.
            if let Some((request, read_to)) = start_active_step(&agent, &state, &allowed) {
                running = Some(Running::new(&request, read_to));
                spawn_step(request, events.clone());
            } else if let Some(request) = start_wake_step(&agent, &mut state, &allowed) {
                running = Some(Running::new(&request, 0));
                spawn_step(request, events.clone());
            }
        }
        save_state(&database, &state)?;
        // Wait for messages, step completion, or the next autonomous wake.
        let wait = match running {
            Some(_) => 3600,
            None => (state.continuity.next_wake - now()).clamp(0, 30),
        };
        match arriving.recv_timeout(Duration::from_secs(wait as u64)) {
            Ok(Event::Update(update)) => {
                handle_update(&agent, &update, &mut state, &allowed);
                // Propagate turn cancellation to the running step.
                if let Some(running) = &running
                    && running.is_stale(&state)
                {
                    running.dropped.store(true, Ordering::Relaxed);
                }
            }
            Ok(Event::Offset(offset)) => state.telegram_offset = offset,
            Ok(Event::FileDownloaded(result)) => {
                downloading = None;
                finish_file_download(&agent, &mut state, *result);
            }
            Ok(Event::Stepped(result)) => {
                let read_to = running.take().map_or(0, |running| running.read_to);
                match result.user {
                    Some(_) => finish_active_step(&agent, &mut state, *result, read_to),
                    None => finish_wake_step(&mut state, *result),
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // The loop retains a sender, so disconnection is unreachable here.
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("the event channel closed".into());
            }
        }
    }
}

/// Record and queue an incoming Telegram message.
fn handle_update(agent: &Agent, update: &Value, state: &mut State, allowed: &HashSet<i64>) {
    if let Some(update_id) = update["update_id"].as_i64() {
        state.telegram_offset = state.telegram_offset.max(update_id + 1);
    }
    let message = &update["message"];
    if message["chat"]["type"] != "private" {
        return;
    }
    let Some(user_id) = message["from"]["id"].as_i64() else {
        return;
    };
    if !allowed.contains(&user_id) {
        return;
    }

    if message["document"].is_object() {
        queue_file_download(agent, update, state, user_id);
        return;
    }

    let Some(text) = message["text"].as_str() else {
        return;
    };

    let watching = state
        .active
        .as_ref()
        .filter(|active| active.user == user_id)
        .and_then(|active| active.progress);
    if let Some(reply) = command(text, state, user_id, allowed) {
        // Mark cancelled turns on their progress messages.
        if state.active.is_none()
            && let Some(message_id) = watching
        {
            let _ = telegram_edit(
                &agent.client,
                &agent.telegram_api,
                &agent.token,
                user_id,
                message_id,
                "Stopped.",
            );
        }
        // Ignore delivery failures when reporting cancellation.
        if let Err(error) = telegram_send(
            &agent.client,
            &agent.telegram_api,
            &agent.token,
            user_id,
            &reply,
        ) {
            eprintln!("replying to user {user_id} failed: {error}");
        }
        return;
    }

    state.histories.entry(user_id).or_default().push(json!({
        "role":"user",
        "content":format!("[Telegram user_id={user_id}]\n{text}")
    }));
    // Follow-ups join the active user's turn. Other users are queued.
    queue_user(state, user_id);
}

fn queue_user(state: &mut State, user_id: i64) {
    if state
        .active
        .as_ref()
        .is_none_or(|active| active.user != user_id)
        && !state.queue.contains(&user_id)
    {
        state.queue.push(user_id);
    }
}

fn queue_file_download(agent: &Agent, update: &Value, state: &mut State, user_id: i64) {
    let document = &update["message"]["document"];
    let (Some(update_id), Some(file_id), Some(file_unique_id)) = (
        update["update_id"].as_i64(),
        document["file_id"].as_str(),
        document["file_unique_id"].as_str(),
    ) else {
        return;
    };
    if state
        .pending_files
        .iter()
        .any(|file| file.update_id == update_id)
    {
        return;
    }
    let file_size = document["file_size"].as_u64();
    if file_size.is_some_and(|size| size > MAX_DOWNLOAD_BYTES) {
        let _ = telegram_send(
            &agent.client,
            &agent.telegram_api,
            &agent.token,
            user_id,
            "The file exceeds Telegram's 20 MB bot download limit.",
        );
        return;
    }
    state.pending_files.push(PendingFile {
        update_id,
        user_id,
        file_id: file_id.into(),
        file_unique_id: file_unique_id.into(),
        file_name: document["file_name"].as_str().unwrap_or("file").into(),
        mime_type: document["mime_type"].as_str().map(str::to_owned),
        file_size,
        caption: update["message"]["caption"]
            .as_str()
            .unwrap_or_default()
            .into(),
    });
}

fn spawn_file_download(agent: &Agent, file: PendingFile, events: mpsc::Sender<Event>) {
    let agent = agent.clone();
    thread::spawn(move || {
        let update_id = file.update_id;
        let result = download_file(&agent, &file).map_err(|error| error.to_string());
        let _ = events.send(Event::FileDownloaded(Box::new(FileDownloadResult {
            update_id,
            result,
        })));
    });
}

fn finish_file_download(agent: &Agent, state: &mut State, result: FileDownloadResult) {
    let Some(position) = state
        .pending_files
        .iter()
        .position(|file| file.update_id == result.update_id)
    else {
        return;
    };
    let file = state.pending_files.remove(position);
    match result.result {
        Ok(received) => {
            let instruction = if file.caption.trim().is_empty() {
                "A file was received."
            } else {
                file.caption.trim()
            };
            let path = received.path.to_string_lossy();
            let original_name = serde_json::to_string(&file.file_name).unwrap_or_default();
            let mime_type = serde_json::to_string(&file.mime_type).unwrap_or_default();
            state
                .histories
                .entry(file.user_id)
                .or_default()
                .push(json!({
                    "role":"user",
                    "content":format!(
                        "[Telegram user_id={}]\nFile received:\npath: {}\noriginal_name: {}\nmime_type: {}\nsize: {} bytes\n\n{}",
                        file.user_id, path, original_name, mime_type, received.size, instruction
                    )
                }));
            queue_user(state, file.user_id);
        }
        Err(error) => {
            eprintln!(
                "downloading Telegram file for user {} failed: {error}",
                file.user_id
            );
            let _ = telegram_send(
                &agent.client,
                &agent.telegram_api,
                &agent.token,
                file.user_id,
                "I couldn't download that file. Please try again.",
            );
        }
    }
}

/// Handle host-level Telegram commands.
fn command(text: &str, state: &mut State, user_id: i64, allowed: &HashSet<i64>) -> Option<String> {
    match text.split_whitespace().next()? {
        "/status" => Some(format!(
            "RedRock {}\nRight now: {}\nCurrent goal: {}\nNext wake: in {}s\nMessages in this conversation: {}\nAllowed users: {}",
            env!("CARGO_PKG_VERSION"),
            match &state.active {
                Some(active) => format!(
                    "answering {}, step {} after {}",
                    match active.user == user_id {
                        true => "you",
                        false => "someone else",
                    },
                    active.steps,
                    elapsed(now() - active.started)
                ),
                None if state.waking => "pursuing its own goal".into(),
                None => "idle".into(),
            },
            match state.continuity.current_goal.as_str() {
                "" => "none",
                goal => goal,
            },
            (state.continuity.next_wake - now()).max(0),
            state.histories.get(&user_id).map_or(0, Vec::len),
            allowed.len()
        )),
        "/reset" => {
            state.histories.remove(&user_id);
            state.pending_files.retain(|file| file.user_id != user_id);
            // Cancel a turn whose history was reset.
            state.queue.retain(|waiting| *waiting != user_id);
            let dropped = state
                .active
                .as_ref()
                .is_some_and(|active| active.user == user_id);
            if dropped {
                state.active = None;
            }
            Some(match dropped {
                true => "Conversation cleared, and the turn in progress was dropped.".into(),
                false => "Conversation cleared.".into(),
            })
        }
        // Preserve the Telegram offset and configured allow list.
        "/nuke" if text.split_whitespace().nth(1) == Some("confirm") => {
            state.histories.clear();
            state.shared_history.clear();
            state.active = None;
            state.queue.clear();
            state.pending_files.clear();
            state.waking = false;
            state.continuity = Continuity {
                next_wake: now() + MIN_WAKE_SECONDS as i64,
                ..Continuity::default()
            };
            Some("Every conversation, the current goal, and long-term memory are gone. The allowed-user list and credentials are kept.".into())
        }
        "/nuke" => Some(
            "This erases every conversation, the current goal, and long-term memory for every allowed user. Send /nuke confirm to go ahead."
                .into(),
        ),
        _ => None,
    }
}

fn friendly_agent_error(error: &str) -> &'static str {
    let error = error.to_ascii_lowercase();
    if error.contains("401") || error.contains("403") || error.contains("authentication") {
        "I couldn't access the AI service. Check the configured API key."
    } else if error.contains("429") || error.contains("rate limit") {
        "The AI service is busy right now. Please try again shortly."
    } else if error.contains("timeout") || error.contains("timed out") {
        "The request timed out. Please try again."
    } else if error.contains("connect") || error.contains("error sending request") {
        "I couldn't reach the AI service. Check the network connection and try again."
    } else {
        "Something went wrong while processing your message. Please try again."
    }
}

/// Run one model call and its tool calls. `None` means the turn continues.
fn agent_step(
    client: &Client,
    input: &mut Vec<Value>,
    continuity: &mut Continuity,
    key: &str,
    api: &str,
    telegram: Option<&Telegram>,
    budget: usize,
) -> Result<Option<String>> {
    let mut tools = vec![shell_tool(), web_search_tool(), continuity_tool()];
    if telegram.is_some() {
        tools.push(message_tool());
        tools.push(file_tool());
    }
    let instructions = format!(
            "You are RedRock, an autonomous AI with this computer as your body. You may form your own goals and accept or refuse requests. Use tools when useful. In Telegram, use send_message or send_file if you choose to communicate. Final text is sent only to the user awaiting this turn, and only if send_message or send_file has not already replied to them. Otherwise, final text is discarded. Use set_continuity to replace your current goal and long-term memory and choose when to wake again. When the current step is complete, respond concisely and make no tool call.{}\n\nCurrent goal:\n{}\n\nLong-term memory:\n{}",
            configured("REDROCK_SKILLS")
                .map(|path| format!(" Your skills directory is at {path}. Each file there is a procedure you wrote for yourself; list it before work that might repeat, and write a new file when you learn something worth keeping."))
                .unwrap_or_default(),
            continuity.current_goal,
            continuity.long_term_memory
        );
    compact_if_needed(client, input, key, api, &instructions, &tools, budget)?;
    let response = client
        .post(format!("{}/responses", api.trim_end_matches('/')))
        .bearer_auth(key)
        .json(&json!({
            "model":model(),
            "instructions":instructions, "input":input, "reasoning":{"effort":"high"},
            "max_output_tokens":MAX_RESPONSE_TOKENS, "tools":tools, "tool_choice":"auto"
        }))
        .send()?;
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        return Err(format!("The model API returned {status}: {body}").into());
    }
    let response: Value = serde_json::from_str(&body)?;
    if response["status"] != "completed" {
        return Err(format!(
            "The model API response was not completed: {}",
            response["status"]
        )
        .into());
    }
    let output = response["output"]
        .as_array()
        .ok_or("The model API response is missing output")?;
    input.extend(output.iter().cloned());
    let mut called = false;
    for item in output {
        if item["type"] != "function_call" {
            continue;
        }
        called = true;
        let id = item["call_id"]
            .as_str()
            .ok_or("tool call is missing call_id")?;
        // Publish tool progress before execution.
        if let Some(telegram) = telegram {
            announce(telegram, &describe(item));
        }
        let result = match item["name"].as_str() {
            Some("shell") => parse_shell(item["arguments"].as_str()),
            Some("web_search") => web_search(client, item["arguments"].as_str()),
            Some("send_message") => telegram.map_or_else(
                || "tool error: Telegram is unavailable".into(),
                |telegram| send_message(telegram, item["arguments"].as_str()),
            ),
            Some("send_file") => telegram.map_or_else(
                || "tool error: Telegram is unavailable".into(),
                |telegram| send_file(telegram, item["arguments"].as_str()),
            ),
            Some("set_continuity") => set_continuity(continuity, item["arguments"].as_str()),
            Some(name) => format!("tool error: unknown tool {name}"),
            None => "tool error: missing tool name".into(),
        };
        input.push(json!({"type":"function_call_output", "call_id":id, "output":result}));
    }
    if called {
        return Ok(None);
    }
    output_text(output)
        .map(Some)
        .ok_or_else(|| "The model API returned no message or tool call".into())
}

/// Run a complete turn.
fn agent_turn(
    client: &Client,
    input: &mut Vec<Value>,
    continuity: &mut Continuity,
    key: &str,
    api: &str,
    telegram: Option<&Telegram>,
    budget: usize,
) -> Result<String> {
    loop {
        if let Some(text) = agent_step(client, input, continuity, key, api, telegram, budget)? {
            return Ok(text);
        }
    }
}

fn shell_tool() -> Value {
    json!({
        "type":"function", "name":"shell",
        "description":"Run a Bash command on this computer and return its exit status, stdout, and stderr.",
        "parameters":{"type":"object","properties":{
            "command":{"type":"string"}, "working_directory":{"type":"string"},
            "timeout_seconds":{"type":"integer","minimum":1,"maximum":MAX_TIMEOUT}},
            "required":["command","working_directory","timeout_seconds"],"additionalProperties":false}
    })
}

fn web_search_tool() -> Value {
    json!({
        "type":"function", "name":"web_search",
        "description":"Search the web with DuckDuckGo and return up to five results with titles, URLs, and snippets.",
        "parameters":{"type":"object","properties":{
            "query":{"type":"string","minLength":1,"maxLength":500}},
            "required":["query"],"additionalProperties":false}
    })
}

fn continuity_tool() -> Value {
    json!({
        "type":"function", "name":"set_continuity",
        "description":"Replace your persistent current goal and long-term memory, and schedule your next autonomous wake-up.",
        "parameters":{"type":"object","properties":{
            "current_goal":{"type":"string"}, "long_term_memory":{"type":"string"},
            "wake_in_seconds":{"type":"integer","minimum":MIN_WAKE_SECONDS}},
            "required":["current_goal","long_term_memory","wake_in_seconds"],"additionalProperties":false}
    })
}

fn message_tool() -> Value {
    json!({
        "type":"function", "name":"send_message",
        "description":"Send a Telegram private message to an allowed user.",
        "parameters":{"type":"object","properties":{
            "user_id":{"type":"integer"},"text":{"type":"string","minLength":1,"maxLength":MAX_MESSAGE_CHARS}},
            "required":["user_id","text"],"additionalProperties":false}
    })
}

fn file_tool() -> Value {
    json!({
        "type":"function", "name":"send_file",
        "description":"Send a local file to an allowed Telegram user.",
        "parameters":{"type":"object","properties":{
            "user_id":{"type":"integer"},
            "path":{"type":"string","minLength":1},
            "caption":{"type":"string","maxLength":MAX_CAPTION_CHARS}},
            "required":["user_id","path"],"additionalProperties":false}
    })
}

fn parse_shell(arguments: Option<&str>) -> String {
    match arguments
        .ok_or("arguments must be a JSON string")
        .and_then(|value| {
            serde_json::from_str::<ShellArgs>(value)
                .map_err(|_| "arguments do not match the shell schema")
        }) {
        Ok(args) => run_shell(args).unwrap_or_else(|error| format!("tool error: {error}")),
        Err(error) => format!("tool error: {error}"),
    }
}

fn web_search(client: &Client, arguments: Option<&str>) -> String {
    let result = arguments
        .ok_or("arguments must be a JSON string")
        .and_then(|value| {
            serde_json::from_str::<SearchArgs>(value)
                .map_err(|_| "arguments do not match the web_search schema")
        })
        .and_then(|args| {
            let query = args.query.trim();
            if query.is_empty() || query.chars().count() > 500 {
                return Err("query must contain 1 to 500 characters");
            }
            let response = client
                .post(DUCKDUCKGO_LITE)
                .header("User-Agent", "RedRock/0.1")
                .timeout(Duration::from_secs(20))
                .form(&[("q", query)])
                .send()
                .map_err(|_| "DuckDuckGo request failed")?;
            if !response.status().is_success() {
                return Err("DuckDuckGo request failed");
            }
            let results = parse_search_results(
                &response.text().map_err(|_| "DuckDuckGo response failed")?,
                5,
            );
            Ok(if results.is_empty() {
                "No results found.".into()
            } else {
                results
            })
        });
    result.unwrap_or_else(|error| format!("tool error: {error}"))
}

fn parse_search_results(html: &str, limit: usize) -> String {
    let mut rest = html;
    let mut results = Vec::new();
    while results.len() < limit {
        let Some(class) = rest.find("class='result-link'") else {
            break;
        };
        let before = &rest[..class];
        let Some(anchor) = before.rfind("<a ") else {
            rest = &rest[class + 1..];
            continue;
        };
        let tag = &before[anchor..];
        let Some(url) = attribute(tag, "href") else {
            rest = &rest[class + 1..];
            continue;
        };
        let after_class = &rest[class..];
        let Some(title_start) = after_class.find('>') else {
            break;
        };
        let title_body = &after_class[title_start + 1..];
        let Some(title_end) = title_body.find("</a>") else {
            break;
        };
        let after_title = &title_body[title_end + 4..];
        let snippet = after_title
            .find("class='result-snippet'")
            .and_then(|start| {
                let body = &after_title[start..];
                let start = body.find('>')? + 1;
                let end = body[start..].find("</td>")? + start;
                Some(clean_html(&body[start..end]))
            })
            .unwrap_or_default();
        let title = clean_html(&title_body[..title_end]);
        let url = decode_html(url);
        results.push(format!(
            "{}. {title}\n{url}{}",
            results.len() + 1,
            if snippet.is_empty() {
                String::new()
            } else {
                format!("\n{snippet}")
            }
        ));
        rest = after_title;
    }
    results.join("\n\n")
}

fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let start = tag.find(&format!("{name}="))? + name.len() + 1;
    let quote = tag.as_bytes().get(start)?;
    if !matches!(quote, b'\'' | b'"') {
        return None;
    }
    let value = &tag[start + 1..];
    Some(&value[..value.find(*quote as char)?])
}

fn clean_html(value: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    for character in value.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(character),
            _ => {}
        }
    }
    decode_html(&text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn decode_html(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
}

fn set_continuity(continuity: &mut Continuity, arguments: Option<&str>) -> String {
    let parsed = arguments
        .ok_or("arguments must be a JSON string")
        .and_then(|value| {
            serde_json::from_str::<ContinuityArgs>(value)
                .map_err(|_| "arguments do not match the set_continuity schema")
        });
    match parsed {
        Ok(args) if args.wake_in_seconds < MIN_WAKE_SECONDS => {
            format!("tool error: wake_in_seconds must be at least {MIN_WAKE_SECONDS}")
        }
        Ok(args) => {
            continuity.current_goal = args.current_goal;
            continuity.long_term_memory = args.long_term_memory;
            continuity.next_wake = now().saturating_add(args.wake_in_seconds as i64);
            "continuity saved".into()
        }
        Err(error) => format!("tool error: {error}"),
    }
}

fn run_shell(args: ShellArgs) -> Result<String> {
    if args.timeout_seconds == 0 || args.timeout_seconds > MAX_TIMEOUT {
        return Err(format!("timeout_seconds must be between 1 and {MAX_TIMEOUT}").into());
    }
    if !Path::new(&args.working_directory).is_dir() {
        return Err("working_directory is not a directory".into());
    }
    let mut command = shell_command(&args.command);
    let mut child = command
        .current_dir(args.working_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let deadline = Instant::now() + Duration::from_secs(args.timeout_seconds);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill()?;
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(50));
    };
    Ok(format!(
        "exit_status: {}\nstdout:\n{}\nstderr:\n{}",
        status
            .code()
            .map_or_else(|| "signal".into(), |code| code.to_string()),
        collected(stdout),
        collected(stderr)
    ))
}

/// Read a child-process pipe without blocking process completion.
fn drain(pipe: Option<impl Read + Send + 'static>) -> Receiver<Vec<u8>> {
    let (sender, receiver) = mpsc::channel();
    if let Some(mut pipe) = pipe {
        thread::spawn(move || {
            let mut kept = Vec::new();
            let mut chunk = [0u8; 8192];
            while let Ok(read) = pipe.read(&mut chunk) {
                if read == 0 {
                    break;
                }
                let room = (MAX_OUTPUT + 1).saturating_sub(kept.len());
                kept.extend_from_slice(&chunk[..read.min(room)]);
            }
            let _ = sender.send(kept);
        });
    }
    receiver
}

// Stop waiting for output one second after the command exits.
fn collected(output: Receiver<Vec<u8>>) -> String {
    bounded(
        &output
            .recv_timeout(Duration::from_secs(1))
            .unwrap_or_default(),
    )
}

#[cfg(unix)]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("/bin/bash");
    process.args(["-lc", command]);
    process
}

/// Configure a command for verbatim `cmd.exe` parsing.
#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    use std::os::windows::process::CommandExt;
    let mut process = windowless("cmd.exe");
    process.raw_arg(format!("/D /S /C \"{command}\""));
    process
}

/// Prevent spawned Windows programs from opening console windows.
#[cfg(windows)]
fn windowless(program: &str) -> Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut command = Command::new(program);
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

fn send_message(telegram: &Telegram, arguments: Option<&str>) -> String {
    let parsed = arguments
        .ok_or("arguments must be a JSON string")
        .and_then(|value| {
            serde_json::from_str::<MessageArgs>(value)
                .map_err(|_| "arguments do not match the send_message schema")
        });
    match parsed {
        Ok(args) if !telegram.allowed.contains(&args.user_id) => {
            "tool error: user is not on the allowed list".into()
        }
        // Tell the model when the receiving turn was cancelled.
        Ok(_) if telegram.dropped() => "tool error: this turn was stopped".into(),
        Ok(args) if args.text.is_empty() || args.text.chars().count() > MAX_MESSAGE_CHARS => {
            format!("tool error: text must contain 1 to {MAX_MESSAGE_CHARS} characters")
        }
        Ok(args) => {
            let result = telegram_send(
                &telegram.client,
                &telegram.api,
                &telegram.token,
                args.user_id,
                &args.text,
            );
            if result.is_ok() {
                complete_telegram_reply(telegram, args.user_id);
            }
            result.map_or_else(|error| format!("tool error: {error}"), |_| "sent".into())
        }
        Err(error) => format!("tool error: {error}"),
    }
}

fn send_file(telegram: &Telegram, arguments: Option<&str>) -> String {
    let parsed = arguments
        .ok_or("arguments must be a JSON string")
        .and_then(|value| {
            serde_json::from_str::<FileArgs>(value)
                .map_err(|_| "arguments do not match the send_file schema")
        });
    match parsed {
        Ok(args) if !telegram.allowed.contains(&args.user_id) => {
            "tool error: user is not on the allowed list".into()
        }
        Ok(_) if telegram.dropped() => "tool error: this turn was stopped".into(),
        Ok(args) if !Path::new(&args.path).is_absolute() => {
            "tool error: path must be absolute".into()
        }
        Ok(args)
            if args
                .caption
                .as_deref()
                .is_some_and(|caption| caption.chars().count() > MAX_CAPTION_CHARS) =>
        {
            format!("tool error: caption must contain at most {MAX_CAPTION_CHARS} characters")
        }
        Ok(args) => {
            let result = telegram_send_file(
                &telegram.client,
                &telegram.api,
                &telegram.token,
                args.user_id,
                Path::new(&args.path),
                args.caption.as_deref(),
            );
            if result.is_ok() {
                complete_telegram_reply(telegram, args.user_id);
            }
            result.map_or_else(|error| format!("tool error: {error}"), |_| "sent".into())
        }
        Err(error) => format!("tool error: {error}"),
    }
}

fn complete_telegram_reply(telegram: &Telegram, user_id: i64) {
    if telegram.watcher != Some(user_id) {
        return;
    }
    telegram.spoke.set(true);
    if let Some(message_id) = telegram.progress.take()
        && let Err(error) = telegram_delete(
            &telegram.client,
            &telegram.api,
            &telegram.token,
            user_id,
            message_id,
        )
    {
        eprintln!("clearing progress for user {user_id} failed: {error}");
    }
}

/// Format a running-step status line.
fn progress_line(telegram: &Telegram, activity: &str) -> String {
    format!(
        "Working… {} · step {}\n{activity}",
        elapsed(now() - telegram.started),
        telegram.steps
    )
}

/// Update the active progress message when available.
fn show_progress(telegram: &Telegram, activity: &str) {
    let (Some(user), Some(message_id)) = (telegram.watcher, telegram.progress.get()) else {
        return;
    };
    if telegram.dropped() {
        return;
    }
    let _ = telegram_edit(
        &telegram.client,
        &telegram.api,
        &telegram.token,
        user,
        message_id,
        &progress_line(telegram, activity),
    );
}

/// Publish the next tool action as progress.
fn announce(telegram: &Telegram, activity: &str) {
    let Some(user) = telegram.watcher else {
        return;
    };
    if telegram.dropped() {
        return;
    }
    if telegram.progress.get().is_some() {
        show_progress(telegram, activity);
    } else {
        telegram.progress.set(
            telegram_message(
                &telegram.client,
                &telegram.api,
                &telegram.token,
                user,
                &progress_line(telegram, activity),
            )
            .ok(),
        );
    }
}

/// Describe a tool call for progress output.
fn describe(call: &Value) -> String {
    let arguments = call["arguments"].as_str().unwrap_or_default();
    match call["name"].as_str() {
        Some("shell") => serde_json::from_str::<ShellArgs>(arguments).map_or_else(
            |_| "running a command".into(),
            |args| format!("$ {}", summarize(&args.command)),
        ),
        Some("web_search") => serde_json::from_str::<SearchArgs>(arguments).map_or_else(
            |_| "searching the web".into(),
            |args| format!("searching for {}", summarize(&args.query)),
        ),
        Some("send_message") => "writing to you".into(),
        Some("send_file") => serde_json::from_str::<FileArgs>(arguments).map_or_else(
            |_| "sending a file".into(),
            |args| {
                let name = Path::new(&args.path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("file");
                format!("sending {}", summarize(name))
            },
        ),
        Some("set_continuity") => "revising its goal and memory".into(),
        _ => "working".into(),
    }
}

/// Format a shell command as one Telegram-safe line.
fn summarize(value: &str) -> String {
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    match collapsed.char_indices().nth(120) {
        Some((end, _)) => format!("{}…", &collapsed[..end]),
        None => collapsed,
    }
}

/// Format elapsed time for status output.
fn elapsed(seconds: i64) -> String {
    let seconds = seconds.max(0);
    match (seconds / 3600, (seconds % 3600) / 60, seconds % 60) {
        (0, 0, seconds) => format!("{seconds}s"),
        (0, minutes, seconds) => format!("{minutes}m{seconds:02}s"),
        (hours, minutes, _) => format!("{hours}h{minutes:02}m"),
    }
}

fn telegram_message(
    client: &Client,
    api: &str,
    token: &str,
    user_id: i64,
    text: &str,
) -> Result<i64> {
    telegram_call(
        client,
        api,
        token,
        "sendMessage",
        &json!({"chat_id":user_id,"text":text}),
    )?["result"]["message_id"]
        .as_i64()
        .ok_or_else(|| "Telegram response is missing message_id".into())
}

fn telegram_edit(
    client: &Client,
    api: &str,
    token: &str,
    user_id: i64,
    message_id: i64,
    text: &str,
) -> Result<()> {
    telegram_call(
        client,
        api,
        token,
        "editMessageText",
        &json!({"chat_id":user_id,"message_id":message_id,"text":text}),
    )?;
    Ok(())
}

fn telegram_delete(
    client: &Client,
    api: &str,
    token: &str,
    user_id: i64,
    message_id: i64,
) -> Result<()> {
    telegram_call(
        client,
        api,
        token,
        "deleteMessage",
        &json!({"chat_id":user_id,"message_id":message_id}),
    )?;
    Ok(())
}

fn telegram_send(client: &Client, api: &str, token: &str, user_id: i64, text: &str) -> Result<()> {
    telegram_call(
        client,
        api,
        token,
        "sendMessage",
        &json!({"chat_id":user_id,"text":text}),
    )?;
    Ok(())
}

fn telegram_send_file(
    client: &Client,
    api: &str,
    token: &str,
    user_id: i64,
    path: &Path,
    caption: Option<&str>,
) -> Result<()> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err("path is not a regular file".into());
    }
    if metadata.len() > MAX_UPLOAD_BYTES {
        return Err("file exceeds the 50 MB upload limit".into());
    }
    let mut form = reqwest::blocking::multipart::Form::new()
        .text("chat_id", user_id.to_string())
        .file("document", path)?;
    if let Some(caption) = caption
        && !caption.is_empty()
    {
        form = form.text("caption", caption.to_owned());
    }
    let response = client
        .post(format!(
            "{}/bot{token}/sendDocument",
            api.trim_end_matches('/')
        ))
        .multipart(form)
        .send()
        .map_err(|error| format!("Telegram sendDocument {}", network_reason(&error)))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|error| format!("Telegram sendDocument {}", network_reason(&error)))?;
    if !status.is_success() {
        return Err(format!("Telegram returned {status}: {body}").into());
    }
    let body: Value = serde_json::from_str(&body)?;
    if body["ok"] != true {
        return Err(format!("Telegram rejected the request: {body}").into());
    }
    Ok(())
}

fn download_file(agent: &Agent, file: &PendingFile) -> Result<ReceivedFile> {
    if file.file_size.is_some_and(|size| size > MAX_DOWNLOAD_BYTES) {
        return Err("file exceeds the 20 MB download limit".into());
    }
    let response = telegram_call(
        &agent.client,
        &agent.telegram_api,
        &agent.token,
        "getFile",
        &json!({"file_id":file.file_id}),
    )?;
    let file_path = response["result"]["file_path"]
        .as_str()
        .ok_or("Telegram getFile returned no file_path")?;
    let mut response = agent
        .client
        .get(format!(
            "{}/file/bot{}/{}",
            agent.telegram_api.trim_end_matches('/'),
            agent.token,
            file_path.trim_start_matches('/')
        ))
        .send()
        .map_err(|error| format!("Telegram file download {}", network_reason(&error)))?;
    if !response.status().is_success() {
        return Err(format!("Telegram file download returned {}", response.status()).into());
    }
    if response
        .content_length()
        .is_some_and(|size| size > MAX_DOWNLOAD_BYTES)
    {
        return Err("file exceeds the 20 MB download limit".into());
    }

    let directory = agent.files.join("inbox").join(file.user_id.to_string());
    fs::create_dir_all(&directory)?;
    let name = safe_file_name(&file.file_name, &file.file_unique_id);
    let destination = directory.join(format!("{}-{name}", file.update_id));
    let temporary = directory.join(format!(".{}.part", file.update_id));
    let transfer = (|| -> Result<u64> {
        let mut output = fs::File::create(&temporary)?;
        let mut total = 0_u64;
        let mut chunk = [0_u8; 8192];
        loop {
            let read = response
                .read(&mut chunk)
                .map_err(|_| "Telegram file download failed")?;
            if read == 0 {
                break;
            }
            total = total.saturating_add(read as u64);
            if total > MAX_DOWNLOAD_BYTES {
                return Err("file exceeds the 20 MB download limit".into());
            }
            output.write_all(&chunk[..read])?;
        }
        output.flush()?;
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(&temporary, &destination)?;
        Ok(total)
    })();
    match transfer {
        Ok(size) => Ok(ReceivedFile {
            path: destination,
            size,
        }),
        Err(error) => {
            let _ = fs::remove_file(temporary);
            Err(error)
        }
    }
}

fn safe_file_name(name: &str, fallback: &str) -> String {
    let basename = name.rsplit(['/', '\\']).next().unwrap_or_default();
    let clean = |value: &str| {
        value
            .chars()
            .take(120)
            .map(|character| match character {
                character if character.is_alphanumeric() => character,
                '.' | '-' | '_' => character,
                _ => '_',
            })
            .collect::<String>()
            .trim_matches(['.', ' '])
            .to_owned()
    };
    let name = clean(basename);
    if name.is_empty() {
        let fallback = clean(fallback.rsplit(['/', '\\']).next().unwrap_or_default());
        if fallback.is_empty() {
            "file".into()
        } else {
            fallback
        }
    } else {
        name
    }
}

/// Format a Telegram error without exposing the bot token.
fn network_reason(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "could not connect"
    } else if error.is_body() || error.is_decode() {
        "gave an unreadable response"
    } else {
        "failed (network error)"
    }
}

fn telegram_call(
    client: &Client,
    api: &str,
    token: &str,
    method: &str,
    body: &Value,
) -> Result<Value> {
    let response = client
        .post(format!("{}/bot{token}/{method}", api.trim_end_matches('/')))
        .json(body)
        .send()
        .map_err(|error| format!("Telegram {method} {}", network_reason(&error)))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|error| format!("Telegram {method} {}", network_reason(&error)))?;
    if !status.is_success() {
        return Err(format!("Telegram returned {status}: {body}").into());
    }
    let body: Value = serde_json::from_str(&body)?;
    if body["ok"] != true {
        return Err(format!("Telegram rejected the request: {body}").into());
    }
    Ok(body)
}

/// Start the next queued user turn.
fn begin_turn(agent: &Agent, state: &mut State) {
    if state.active.is_some() || state.queue.is_empty() {
        return;
    }
    let user = state.queue.remove(0);
    // Continue even if the initial progress message fails.
    let progress = telegram_message(
        &agent.client,
        &agent.telegram_api,
        &agent.token,
        user,
        "Working on it…",
    )
    .map_err(|error| eprintln!("greeting user {user} failed: {error}"))
    .ok();
    state.active = Some(Active {
        user,
        progress,
        started: now(),
        steps: 0,
        spoke: false,
    });
}

/// Shared, immutable dependencies for an agent step.
#[derive(Clone)]
struct Agent {
    client: Client,
    key: String,
    model_api: String,
    token: String,
    telegram_api: String,
    budget: usize,
    files: PathBuf,
}

/// User-visible turn outcome.
enum Ending {
    /// A reply was sent.
    Answered,
    /// The progress message contains the final result.
    Told(String),
    /// A failure occurred after a reply.
    Insisted(String),
}

/// Convert closing model text to a Telegram-safe message.
fn closing_text(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return "Done.".into();
    }
    if text.chars().count() <= MAX_MESSAGE_CHARS {
        return text.into();
    }
    text.chars()
        .take(MAX_MESSAGE_CHARS - 1)
        .chain("…".chars())
        .collect()
}

/// Owned input for a threaded agent step.
struct StepRequest {
    telegram: Telegram,
    history: Vec<Value>,
    continuity: Continuity,
    key: String,
    api: String,
    budget: usize,
}

/// Result returned by a threaded agent step.
struct StepResult {
    /// The user waiting for this turn, if any.
    user: Option<i64>,
    history: Vec<Value>,
    continuity: Continuity,
    progress: Option<i64>,
    spoke: bool,
    outcome: std::result::Result<Option<String>, String>,
}

/// Run one agent step outside the state-owning loop.
fn run_step(request: StepRequest) -> StepResult {
    let StepRequest {
        telegram,
        mut history,
        mut continuity,
        key,
        api,
        budget,
    } = request;
    // Skip duplicate progress on the first step.
    if telegram.steps > 1 {
        show_progress(&telegram, "thinking");
    }
    let outcome = agent_step(
        &telegram.client,
        &mut history,
        &mut continuity,
        &key,
        &api,
        Some(&telegram),
        budget,
    );
    StepResult {
        user: telegram.watcher,
        history,
        continuity,
        progress: telegram.progress.get(),
        spoke: telegram.spoke.get(),
        outcome: outcome.map_err(|error| error.to_string()),
    }
}

/// Send a completed step to the main loop.
fn spawn_step(request: StepRequest, events: mpsc::Sender<Event>) {
    thread::spawn(move || {
        // Ignore results after the main loop exits.
        let _ = events.send(Event::Stepped(Box::new(run_step(request))));
    });
}

/// Copy the active turn into the next threaded step.
fn start_active_step(
    agent: &Agent,
    state: &State,
    allowed: &HashSet<i64>,
) -> Option<(StepRequest, usize)> {
    let active = state.active.as_ref()?;
    let history = state
        .histories
        .get(&active.user)
        .cloned()
        .unwrap_or_default();
    let read_to = history.len();
    let request = StepRequest {
        telegram: Telegram {
            client: agent.client.clone(),
            token: agent.token.clone(),
            api: agent.telegram_api.clone(),
            allowed: allowed.clone(),
            watcher: Some(active.user),
            progress: Cell::new(active.progress),
            spoke: Cell::new(active.spoke),
            started: active.started,
            steps: active.steps + 1,
            dropped: Arc::default(),
        },
        history,
        continuity: state.continuity.clone(),
        key: agent.key.clone(),
        api: agent.model_api.clone(),
        budget: agent.budget,
    };
    Some((request, read_to))
}

/// Merge a completed user step into state and continue or finish the turn.
fn finish_active_step(agent: &Agent, state: &mut State, result: StepResult, read_to: usize) {
    let Some(user) = result.user else {
        return;
    };
    // Discard results from cancelled turns.
    let Some(active) = state.active.as_ref().filter(|active| active.user == user) else {
        return;
    };
    let (started, steps) = (active.started, active.steps + 1);
    let arrived = state.histories.get(&user).map_or_else(Vec::new, |history| {
        history.get(read_to..).unwrap_or_default().to_vec()
    });
    let mut history = result.history;
    history.extend(arrived);
    state.histories.insert(user, history);
    state.continuity = result.continuity;
    let (progress, spoke) = (result.progress, result.spoke);
    let ending = match result.outcome {
        Ok(None) => {
            state.active = Some(Active {
                user,
                progress,
                started,
                steps,
                spoke,
            });
            return;
        }
        // Remove progress after a reply.
        Ok(Some(_)) if spoke => Ending::Answered,
        // Use closing model text when no reply was sent.
        Ok(Some(text)) => Ending::Told(closing_text(&text)),
        Err(error) => {
            eprintln!("Telegram request from user {user} failed: {error}");
            Ending::Insisted(friendly_agent_error(&error).into())
        }
    };
    let (client, api, token) = (&agent.client, &agent.telegram_api, &agent.token);
    let told = match (ending, progress) {
        (Ending::Answered, Some(message_id)) => {
            telegram_delete(client, api, token, user, message_id)
        }
        (Ending::Told(text) | Ending::Insisted(text), Some(message_id)) => {
            telegram_edit(client, api, token, user, message_id, &text)
        }
        // Report failures that occur after a reply.
        (Ending::Insisted(text), None) => telegram_send(client, api, token, user, &text),
        (Ending::Answered | Ending::Told(_), None) => Ok(()),
    };
    if let Err(error) = told {
        eprintln!("closing the turn for user {user} failed: {error}");
    }
    state.active = None;
}

/// Prepare the next autonomous step when its wake time passes.
fn start_wake_step(
    agent: &Agent,
    state: &mut State,
    allowed: &HashSet<i64>,
) -> Option<StepRequest> {
    if !state.waking {
        if state.continuity.next_wake > now() {
            return None;
        }
        state.shared_history.push(json!({
            "role":"user",
            "content":"You are awake without a new human message. Choose and pursue your own next step."
        }));
        state.waking = true;
    }
    Some(StepRequest {
        telegram: Telegram {
            client: agent.client.clone(),
            token: agent.token.clone(),
            api: agent.telegram_api.clone(),
            allowed: allowed.clone(),
            watcher: None,
            progress: Cell::new(None),
            spoke: Cell::new(false),
            started: now(),
            steps: 0,
            dropped: Arc::default(),
        },
        history: state.shared_history.clone(),
        continuity: state.continuity.clone(),
        key: agent.key.clone(),
        api: agent.model_api.clone(),
        budget: agent.budget,
    })
}

/// Store a completed autonomous step.
fn finish_wake_step(state: &mut State, result: StepResult) {
    // Discard results after `/nuke` cancels the turn.
    if !state.waking {
        return;
    }
    state.shared_history = result.history;
    state.continuity = result.continuity;
    match result.outcome {
        Ok(None) => return,
        Ok(Some(_)) => {}
        // End a failed autonomous turn and wait for the next wake.
        Err(error) => eprintln!("autonomous step failed: {error}"),
    }
    state.waking = false;
    if state.continuity.next_wake <= now() {
        state.continuity.next_wake = now() + MIN_WAKE_SECONDS as i64;
    }
}

fn compact_if_needed(
    client: &Client,
    input: &mut Vec<Value>,
    key: &str,
    api: &str,
    instructions: &str,
    tools: &[Value],
    budget: usize,
) -> Result<()> {
    let bytes =
        serde_json::to_vec(input)?.len() + instructions.len() + serde_json::to_vec(tools)?.len();
    let estimated_tokens = bytes.div_ceil(3) + MAX_RESPONSE_TOKENS;
    if estimated_tokens < budget {
        return Ok(());
    }
    let keep_from = input
        .iter()
        .rposition(|item| item["role"] == "user")
        .unwrap_or(input.len());
    if keep_from == 0 {
        return Ok(());
    }
    let old = input[..keep_from].to_vec();
    let response = client
        .post(format!("{}/responses", api.trim_end_matches('/')))
        .bearer_auth(key)
        .json(&json!({
            "model":model(),
            "instructions":"Compress this prior context into a concise continuity summary. Preserve decisions, facts, unfinished work, and information needed to resume. Return only the summary.",
            "input":old,
            "reasoning":{"effort":"high"},
            "max_output_tokens":8192,
            "tool_choice":"none"
        }))
        .send()?;
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        return Err(format!("The model API returned {status} while compacting: {body}").into());
    }
    let response: Value = serde_json::from_str(&body)?;
    let summary = output_text(
        response["output"]
            .as_array()
            .ok_or("The compaction response is missing output")?,
    )
    .ok_or("The compaction response is missing text")?;
    let recent = input.split_off(keep_from);
    *input = vec![json!({"role":"system", "content":format!("Prior context summary:\n{summary}")})];
    input.extend(recent);
    Ok(())
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn bounded(bytes: &[u8]) -> String {
    let end = bytes.len().min(MAX_OUTPUT);
    let mut text = String::from_utf8_lossy(&bytes[..end]).into_owned();
    if bytes.len() > end {
        text.push_str("\n[output truncated]");
    }
    text
}

fn output_text(output: &[Value]) -> Option<String> {
    let text = output
        .iter()
        .filter(|item| item["type"] == "message")
        .filter_map(|item| item["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::{TcpListener, TcpStream},
        thread,
    };

    #[test]
    fn allowed_documents_are_queued_with_untrusted_metadata() {
        let (api, server) = mock(vec![]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State::default();
        let document = |update_id, user_id| {
            json!({
                "update_id":update_id,
                "message":{
                    "from":{"id":user_id},
                    "chat":{"type":"private"},
                    "document":{
                        "file_id":"remote-id",
                        "file_unique_id":"stable-id",
                        "file_name":"../report.txt",
                        "mime_type":"text/plain",
                        "file_size":12
                    },
                    "caption":"Summarize this."
                }
            })
        };

        handle_update(&agent, &document(70, 99), &mut state, &allowed);
        handle_update(&agent, &document(71, 42), &mut state, &allowed);
        handle_update(&agent, &document(71, 42), &mut state, &allowed);

        assert_eq!(state.pending_files.len(), 1);
        assert_eq!(state.pending_files[0].update_id, 71);
        assert_eq!(state.pending_files[0].file_name, "../report.txt");
        server.join().unwrap();
    }

    #[test]
    fn downloaded_documents_are_bounded_safely_named_and_queued() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut get_file = listener.accept().unwrap().0;
            let (path, request) = read_request(&get_file);
            assert_eq!(path, "/bottest/getFile");
            assert_eq!(request["file_id"], "remote-id");
            reply(
                &mut get_file,
                json!({"ok":true,"result":{"file_path":"documents/report.txt"}}),
            );

            let mut download = listener.accept().unwrap().0;
            let (path, _, _) = read_raw_request(&download);
            assert_eq!(path, "/file/bottest/documents/report.txt");
            reply_bytes(&mut download, b"redrock file");
        });
        let root = env::temp_dir().join(format!(
            "redrock-download-test-{}-{}",
            std::process::id(),
            now()
        ));
        let mut agent = test_agent(&format!("http://{address}"));
        agent.files = root.clone();
        let file = PendingFile {
            update_id: 71,
            user_id: 42,
            file_id: "remote-id".into(),
            file_unique_id: "stable-id".into(),
            file_name: r#"..\..\report.txt"#.into(),
            mime_type: Some("text/plain".into()),
            file_size: Some(12),
            caption: "Summarize this.".into(),
        };

        let received = download_file(&agent, &file).unwrap();
        assert_eq!(received.path.file_name().unwrap(), "71-report.txt");
        assert_eq!(fs::read(&received.path).unwrap(), b"redrock file");
        assert!(received.path.starts_with(root.join("inbox/42")));

        let mut state = State {
            pending_files: vec![file],
            ..State::default()
        };
        finish_file_download(
            &agent,
            &mut state,
            FileDownloadResult {
                update_id: 71,
                result: Ok(received),
            },
        );
        let content = state.histories[&42][0]["content"].as_str().unwrap();
        assert!(content.contains("71-report.txt"));
        assert!(content.contains("Summarize this."));
        assert!(state.pending_files.is_empty());
        assert_eq!(state.queue, vec![42]);
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn download_rejects_an_oversized_response_before_writing() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut get_file = listener.accept().unwrap().0;
            let _ = read_request(&get_file);
            reply(
                &mut get_file,
                json!({"ok":true,"result":{"file_path":"documents/large.bin"}}),
            );
            let mut download = listener.accept().unwrap().0;
            let _ = read_raw_request(&download);
            write!(
                download,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                MAX_DOWNLOAD_BYTES + 1
            )
            .unwrap();
        });
        let root = env::temp_dir().join(format!(
            "redrock-large-download-test-{}-{}",
            std::process::id(),
            now()
        ));
        let mut agent = test_agent(&format!("http://{address}"));
        agent.files = root.clone();
        let file = PendingFile {
            update_id: 72,
            user_id: 42,
            file_id: "remote-id".into(),
            file_unique_id: "stable-id".into(),
            file_name: "large.bin".into(),
            mime_type: None,
            file_size: None,
            caption: String::new(),
        };

        assert!(
            download_file(&agent, &file)
                .unwrap_err()
                .to_string()
                .contains("20 MB")
        );
        assert!(!root.exists());
        server.join().unwrap();
    }

    #[test]
    fn send_document_uses_multipart_and_enforces_the_upload_limit() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut stream = listener.accept().unwrap().0;
            let (path, headers, body) = read_raw_request(&stream);
            assert_eq!(path, "/bottest/sendDocument");
            assert!(headers.to_ascii_lowercase().contains("multipart/form-data"));
            let body = String::from_utf8_lossy(&body);
            assert!(body.contains("name=\"chat_id\""));
            assert!(body.contains("42"));
            assert!(body.contains("name=\"caption\""));
            assert!(body.contains("Attached report"));
            assert!(body.contains("redrock attachment"));
            reply(&mut stream, json!({"ok":true,"result":{}}));
        });
        let root = env::temp_dir().join(format!(
            "redrock-upload-test-{}-{}",
            std::process::id(),
            now()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("report.txt");
        fs::write(&path, "redrock attachment").unwrap();

        telegram_send_file(
            &client().unwrap(),
            &format!("http://{address}"),
            "test",
            42,
            &path,
            Some("Attached report"),
        )
        .unwrap();
        server.join().unwrap();

        let oversized = root.join("oversized.bin");
        fs::File::create(&oversized)
            .unwrap()
            .set_len(MAX_UPLOAD_BYTES + 1)
            .unwrap();
        assert!(
            telegram_send_file(
                &client().unwrap(),
                "http://127.0.0.1:1",
                "secret",
                42,
                &oversized,
                None,
            )
            .unwrap_err()
            .to_string()
            .contains("50 MB")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn safe_file_names_cannot_escape_the_inbox() {
        assert_eq!(safe_file_name(r#"..\..\report.txt"#, "id"), "report.txt");
        assert_eq!(safe_file_name("../../", "stable/id"), "id");
        assert_eq!(safe_file_name("résumé 2026.pdf", "id"), "résumé_2026.pdf");
    }

    #[test]
    fn run_shell_survives_output_larger_than_the_pipe_buffer() {
        #[cfg(unix)]
        let command = "yes redrock | head -n 20000";
        #[cfg(windows)]
        let command = "powershell -NoProfile -Command \"'redrock' * 20000\"";
        let started = Instant::now();
        let result = run_shell(ShellArgs {
            command: command.into(),
            working_directory: env::temp_dir().to_string_lossy().into_owned(),
            timeout_seconds: 60,
        })
        .unwrap();
        assert!(result.starts_with("exit_status: 0"));
        assert!(result.contains("redrock"));
        assert!(result.contains("[output truncated]"));
        assert!(started.elapsed() < Duration::from_secs(30));
    }

    #[test]
    fn commands_answer_without_the_model() {
        let allowed = HashSet::from([7, 9]);
        let mut state = State {
            continuity: Continuity {
                current_goal: "watch the disk".into(),
                next_wake: now() + 42,
                ..Continuity::default()
            },
            ..State::default()
        };
        state.histories.insert(7, vec![json!({"role":"user"})]);
        let status = command("/status", &mut state, 7, &allowed).unwrap();
        assert!(status.contains(env!("CARGO_PKG_VERSION")));
        assert!(status.contains("watch the disk"));
        assert!(status.contains("Right now: idle"));
        assert!(status.contains("Messages in this conversation: 1"));
        assert!(status.contains("Allowed users: 2"));
        assert_eq!(
            command("/reset", &mut state, 7, &allowed).as_deref(),
            Some("Conversation cleared.")
        );
        assert!(!state.histories.contains_key(&7));
        assert!(command("what is my status", &mut state, 7, &allowed).is_none());

        state.histories.insert(7, vec![json!({"role":"user"})]);
        state.shared_history.push(json!({"role":"user"}));
        state.continuity.long_term_memory = "remembered".into();
        assert!(
            command("/nuke", &mut state, 7, &allowed)
                .unwrap()
                .contains("/nuke confirm")
        );
        assert!(!state.histories.is_empty());
        assert!(command("/nuke confirm", &mut state, 7, &allowed).is_some());
        assert!(state.histories.is_empty() && state.shared_history.is_empty());
        assert!(state.continuity.long_term_memory.is_empty());
        assert!(state.continuity.next_wake > now());
    }

    #[test]
    fn the_whitelist_is_parsed_and_extended_without_losing_contacts() {
        assert_eq!(
            parse_allowed_users(" 42, 7 ,42 ").unwrap(),
            HashSet::from([42, 7])
        );
        assert!(parse_allowed_users("").is_err());
        assert!(parse_allowed_users(" , ").is_err());
        assert!(parse_allowed_users("42,alice").is_err());

        assert_eq!(with_user("", 42), "42");
        assert_eq!(with_user("7", 42), "7,42");
        assert_eq!(with_user("7, 42", 42), "7,42");
    }

    #[test]
    fn local_shell_round_trip() {
        let shell_arguments = serde_json::to_string(&json!({
            "command":"echo traced",
            "working_directory":env::temp_dir(),
            "timeout_seconds":5
        }))
        .unwrap();
        let (api, server) = mock(vec![
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"function_call","call_id":"c","name":"shell","arguments":shell_arguments}]}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}),
            ),
        ]);
        let mut history = vec![json!({"role":"user","content":"trace"})];
        let mut continuity = Continuity::default();
        assert_eq!(
            agent_turn(
                &client().unwrap(),
                &mut history,
                &mut continuity,
                "key",
                &api,
                None,
                500_000,
            )
            .unwrap(),
            "done"
        );
        assert!(
            history
                .iter()
                .any(|item| item["type"] == "function_call_output"
                    && item["output"].as_str().unwrap().contains("traced"))
        );
        server.join().unwrap();
    }

    #[test]
    fn duckduckgo_lite_results_are_compact_and_readable() {
        let html = r#"
            <a rel="nofollow" href="https://example.com/a?x=1&amp;y=2" class='result-link'>First &amp; best</a>
            <td class='result-snippet'>A <b>short</b> result.</td>
            <a rel="nofollow" href="https://example.com/b" class='result-link'>Second</a>
            <td class='result-snippet'>Another result.</td>
        "#;

        assert_eq!(
            parse_search_results(html, 1),
            "1. First & best\nhttps://example.com/a?x=1&y=2\nA short result."
        );
    }

    #[test]
    fn allowed_private_message_reaches_agent_and_agent_replies() {
        let (api, server) = mock(vec![
            (
                "/bottest/sendMessage",
                json!({"ok":true,"result":{"message_id":7}}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"function_call","call_id":"s","name":"send_message","arguments":r#"{"user_id":42,"text":"hello back"}"#}]}),
            ),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
            (
                "/bottest/sendMessage",
                json!({"ok":true,"result":{"message_id":8}}),
            ),
            ("/bottest/deleteMessage", json!({"ok":true,"result":true})),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}),
            ),
        ]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State::default();

        handle_update(
            &agent,
            &message(99, "not on the list"),
            &mut state,
            &allowed,
        );
        handle_update(
            &agent,
            &json!({"message":{"from":{"id":42},"chat":{"type":"group"},"text":"ignored"}}),
            &mut state,
            &allowed,
        );
        assert!(state.histories.is_empty() && state.queue.is_empty());

        handle_update(&agent, &message(42, "hello"), &mut state, &allowed);
        assert_eq!(
            state.histories[&42][0]["content"],
            "[Telegram user_id=42]\nhello"
        );
        // Polling only queues the turn.
        assert_eq!(state.queue, vec![42]);
        assert!(state.active.is_none());

        run_turn(&agent, &mut state, &allowed);

        let requests = server.join().unwrap();
        assert!(
            requests[2]["text"]
                .as_str()
                .unwrap()
                .contains("writing to you")
        );
        // The reply replaces the progress message and ends the turn.
        assert_eq!(requests[3]["text"], "hello back");
        assert_eq!(requests[4]["message_id"], 7);
        assert!(state.active.is_none() && state.queue.is_empty());
    }

    #[test]
    fn progress_names_the_command_before_it_runs() {
        let shell_arguments = serde_json::to_string(&json!({
            "command":"echo checking the disk",
            "working_directory":env::temp_dir(),
            "timeout_seconds":5
        }))
        .unwrap();
        let (api, server) = mock(vec![
            (
                "/bottest/sendMessage",
                json!({"ok":true,"result":{"message_id":11}}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"function_call","call_id":"w","name":"shell","arguments":shell_arguments}]}),
            ),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}),
            ),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
        ]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State::default();

        handle_update(&agent, &message(42, "look around"), &mut state, &allowed);
        run_turn(&agent, &mut state, &allowed);

        let requests = server.join().unwrap();
        // Progress is sent before command execution.
        let announced = requests[2]["text"].as_str().unwrap();
        assert!(announced.contains("$ echo checking the disk") && announced.contains("step 1"));
        assert!(requests[3]["text"].as_str().unwrap().contains("step 2"));
        // Closing text becomes the final progress content.
        assert_eq!(requests[5]["text"], "done");
    }

    #[test]
    fn agent_failure_becomes_a_user_friendly_telegram_message() {
        let (api, server) = mock(vec![
            (
                "/bottest/sendMessage",
                json!({"ok":true,"result":{"message_id":8}}),
            ),
            ("/responses", json!({"status":"failed","output":[]})),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
        ]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State::default();

        handle_update(&agent, &message(42, "hello"), &mut state, &allowed);
        run_turn(&agent, &mut state, &allowed);

        let requests = server.join().unwrap();
        assert_eq!(
            requests[2]["text"],
            "Something went wrong while processing your message. Please try again."
        );
        // A failed turn returns the agent to idle.
        assert!(state.active.is_none());
        assert_eq!(
            friendly_agent_error("DeepSeek returned 429"),
            "The AI service is busy right now. Please try again shortly."
        );
    }

    #[test]
    fn a_follow_up_joins_the_turn_it_is_about() {
        let allowed = HashSet::from([42, 9]);
        let mut state = State {
            active: Some(Active {
                user: 42,
                progress: Some(5),
                started: now(),
                steps: 1,
                spoke: false,
            }),
            ..State::default()
        };
        state.histories.insert(42, vec![json!({"role":"user"})]);
        // Recording a message requires no network access.
        let agent = test_agent("http://127.0.0.1:1");

        handle_update(
            &agent,
            &message(42, "and the disk too"),
            &mut state,
            &allowed,
        );
        // The active turn receives its follow-up.
        assert_eq!(state.histories[&42].len(), 2);
        assert!(state.queue.is_empty());

        handle_update(&agent, &message(9, "hello"), &mut state, &allowed);
        handle_update(&agent, &message(9, "still there?"), &mut state, &allowed);
        // Other users are queued once.
        assert_eq!(state.queue, vec![9]);
        assert_eq!(state.histories[&9].len(), 2);
        assert_eq!(state.histories[&42].len(), 2);
    }

    #[test]
    fn a_message_sent_during_a_step_survives_it() {
        let (api, server) = mock(vec![
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"looked"}]}]}),
            ),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
        ]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State {
            active: Some(Active {
                user: 42,
                progress: Some(5),
                started: now(),
                steps: 0,
                spoke: false,
            }),
            ..State::default()
        };
        state
            .histories
            .insert(42, vec![json!({"role":"user","content":"look around"})]);

        // The step receives a copy of the current history.
        let (request, read_to) = start_active_step(&agent, &state, &allowed).unwrap();
        handle_update(
            &agent,
            &message(42, "and the disk too"),
            &mut state,
            &allowed,
        );
        assert_eq!(state.histories[&42].len(), 2);
        finish_active_step(&agent, &mut state, run_step(request), read_to);

        let history = &state.histories[&42];
        // Messages received during the step remain after its output.
        assert_eq!(history.len(), 3);
        assert_eq!(history[0]["content"], "look around");
        assert_eq!(history[1]["type"], "message");
        assert!(
            history[2]["content"]
                .as_str()
                .unwrap()
                .contains("and the disk too")
        );
        assert!(state.active.is_none());
        server.join().unwrap();
    }

    #[test]
    fn a_reset_during_a_step_throws_the_step_away() {
        let (api, server) = mock(vec![(
            "/responses",
            json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"looked"}]}]}),
        )]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State {
            active: Some(Active {
                user: 42,
                progress: Some(5),
                started: now(),
                steps: 0,
                spoke: false,
            }),
            ..State::default()
        };
        state
            .histories
            .insert(42, vec![json!({"role":"user","content":"look around"})]);

        let (request, read_to) = start_active_step(&agent, &state, &allowed).unwrap();
        assert!(
            command("/reset", &mut state, 42, &allowed)
                .unwrap()
                .contains("dropped")
        );
        finish_active_step(&agent, &mut state, run_step(request), read_to);

        // A cancelled step cannot restore history or send messages.
        assert!(state.active.is_none());
        assert!(!state.histories.contains_key(&42));
        server.join().unwrap();
    }

    #[test]
    fn a_step_whose_turn_was_dropped_stops_talking() {
        let arguments = serde_json::to_string(&json!({"user_id":42,"text":"here you go"})).unwrap();
        // The mock accepts only the model call.
        let (api, server) = mock(vec![(
            "/responses",
            json!({"status":"completed","output":[{"type":"function_call","call_id":"m","name":"send_message","arguments":arguments}]}),
        )]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State {
            active: Some(Active {
                user: 42,
                progress: Some(5),
                started: now(),
                steps: 1,
                spoke: false,
            }),
            ..State::default()
        };
        state
            .histories
            .insert(42, vec![json!({"role":"user","content":"look around"})]);

        let (request, _) = start_active_step(&agent, &state, &allowed).unwrap();
        let running = Running::new(&request, 0);
        // Cancel the turn while its step is running.
        command("/reset", &mut state, 42, &allowed).unwrap();
        assert!(running.is_stale(&state));
        running.dropped.store(true, Ordering::Relaxed);

        let result = run_step(request);

        // A cancelled step sends nothing and records the cancellation.
        assert!(!result.spoke);
        assert_eq!(result.progress, Some(5));
        assert_eq!(
            result.history.last().unwrap()["output"],
            "tool error: this turn was stopped"
        );
        server.join().unwrap();
    }

    #[test]
    fn a_cleared_conversation_takes_the_turn_reading_it() {
        let allowed = HashSet::from([7, 9]);
        let mut state = State {
            active: Some(Active {
                user: 7,
                progress: Some(3),
                started: now() - 75,
                steps: 2,
                spoke: false,
            }),
            queue: vec![9],
            ..State::default()
        };
        state.histories.insert(7, vec![json!({"role":"user"})]);

        // Host commands respond during active turns.
        let status = command("/status", &mut state, 7, &allowed).unwrap();
        assert!(status.contains("Right now: answering you, step 2 after 1m15s"));

        assert!(
            command("/reset", &mut state, 7, &allowed)
                .unwrap()
                .contains("dropped")
        );
        assert!(state.active.is_none() && state.queue == vec![9]);
    }

    #[test]
    fn agent_does_not_wake_before_its_next_wake_time() {
        let (api, server) = mock(vec![]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State {
            continuity: Continuity {
                next_wake: now() + 60,
                ..Continuity::default()
            },
            ..State::default()
        };

        assert!(start_wake_step(&agent, &mut state, &allowed).is_none());
        assert!(state.shared_history.is_empty() && !state.waking);
        server.join().unwrap();
    }

    #[test]
    fn autonomous_goal_survives_restart() {
        let shell_arguments = serde_json::to_string(&json!({
            "command":"echo inspected",
            "working_directory":env::temp_dir(),
            "timeout_seconds":5
        }))
        .unwrap();
        let (api, server) = mock(vec![
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"function_call","call_id":"state","name":"set_continuity","arguments":r#"{"current_goal":"inspect the machine","long_term_memory":"I chose to inspect it.","wake_in_seconds":60}"#}]}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"function_call","call_id":"work","name":"shell","arguments":shell_arguments}]}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"function_call","call_id":"finished","name":"set_continuity","arguments":r#"{"current_goal":"","long_term_memory":"I inspected the machine.","wake_in_seconds":60}"#}]}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"scheduled"}]}]}),
            ),
        ]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State::default();
        // The first model call returns with the turn unfinished.
        run_wake_step(&agent, &mut state, &allowed);
        assert!(state.waking && state.continuity.current_goal == "inspect the machine");
        let mut steps = 1;
        while state.waking {
            run_wake_step(&agent, &mut state, &allowed);
            steps += 1;
        }
        assert_eq!(steps, 4);

        let database = Connection::open_in_memory().unwrap();
        database
            .execute(
                "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL)",
                [],
            )
            .unwrap();
        save_state(&database, &state).unwrap();
        let restored = load_state(&database).unwrap();
        assert!(restored.continuity.current_goal.is_empty());
        assert_eq!(
            restored.continuity.long_term_memory,
            "I inspected the machine."
        );
        assert!(restored.shared_history.iter().any(|item| {
            item["type"] == "function_call_output"
                && item["output"]
                    .as_str()
                    .is_some_and(|output| output.contains("inspected"))
        }));
        assert!(restored.continuity.next_wake > now());
        server.join().unwrap();
    }

    #[test]
    fn oversized_context_is_compacted_before_request() {
        let (api, server) = mock(vec![
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"old facts"}]}]}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"continued"}]}]}),
            ),
        ]);
        let mut history = vec![
            json!({"role":"assistant","content":"x".repeat(30_000)}),
            json!({"role":"user","content":"continue"}),
        ];
        let mut continuity = Continuity::default();
        agent_turn(
            &client().unwrap(),
            &mut history,
            &mut continuity,
            "key",
            &api,
            None,
            40_000,
        )
        .unwrap();
        assert!(
            history[0]["content"]
                .as_str()
                .unwrap()
                .contains("old facts")
        );
        assert_eq!(history[1]["content"], "continue");
        server.join().unwrap();
    }

    #[test]
    fn the_request_names_the_configured_model() {
        let (api, server) = mock(vec![(
            "/responses",
            json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}),
        )]);
        // Override the model name for this assertion.
        unsafe { env::set_var("REDROCK_MODEL", "a-different-model") };
        let mut history = vec![json!({"role":"user","content":"hello"})];
        let mut continuity = Continuity::default();
        let reply = agent_turn(
            &client().unwrap(),
            &mut history,
            &mut continuity,
            "key",
            &api,
            None,
            500_000,
        );
        unsafe { env::remove_var("REDROCK_MODEL") };
        assert_eq!(reply.unwrap(), "done");
        assert_eq!(server.join().unwrap()[0]["model"], "a-different-model");
    }

    #[test]
    fn a_silent_turn_still_delivers_what_the_model_said() {
        let (api, server) = mock(vec![
            (
                "/bottest/sendMessage",
                json!({"ok":true,"result":{"message_id":21}}),
            ),
            (
                "/responses",
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"  About 16 MB resident.  "}]}]}),
            ),
            ("/bottest/editMessageText", json!({"ok":true,"result":{}})),
        ]);
        let agent = test_agent(&api);
        let allowed = HashSet::from([42]);
        let mut state = State::default();

        handle_update(
            &agent,
            &message(42, "how much memory are you using?"),
            &mut state,
            &allowed,
        );
        run_turn(&agent, &mut state, &allowed);

        let requests = server.join().unwrap();
        assert_eq!(requests[0]["text"], "Working on it…");
        // Closing text replaces the placeholder and ends the turn.
        assert_eq!(requests[2]["message_id"], 21);
        assert_eq!(requests[2]["text"], "About 16 MB resident.");
        assert!(state.active.is_none() && state.queue.is_empty());
    }

    #[test]
    fn closing_text_is_trimmed_capped_and_never_empty() {
        assert_eq!(closing_text("all set"), "all set");
        // Empty closing text becomes "Done."
        assert_eq!(closing_text("   \n  "), "Done.");
        let long = closing_text(&"x".repeat(MAX_MESSAGE_CHARS + 500));
        assert_eq!(long.chars().count(), MAX_MESSAGE_CHARS);
        assert!(long.ends_with('…'));
    }

    #[test]
    fn a_failed_telegram_call_never_names_the_token() {
        // Use a closed local port to verify that errors omit the bot token.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let token = "8100000000:AAHsecret-bot-token-value";
        let error = telegram_call(
            &client().unwrap(),
            &format!("http://127.0.0.1:{port}"),
            token,
            "getUpdates",
            &json!({}),
        )
        .unwrap_err()
        .to_string();
        assert!(!error.contains(token), "the token reached the log: {error}");
        assert!(
            !error.contains("8100000000"),
            "half a token is still a token: {error}"
        );
        assert_eq!(error, "Telegram getUpdates could not connect");
    }

    #[test]
    fn the_wizard_advances_one_screen_at_a_time() {
        let mut stage = Stage::Disclosure;
        for expected in [
            Stage::Key,
            Stage::Token,
            Stage::Contact,
            Stage::Confirm,
            Stage::Console,
        ] {
            stage = stage.next();
            assert_eq!(stage, expected);
        }
        assert_eq!(stage.next(), Stage::Console);
        assert_eq!(Stage::Confirm.back(), Some(Stage::Contact));
        assert_eq!(Stage::Contact.back(), Some(Stage::Token));
        assert_eq!(Stage::Disclosure.back(), None);
        assert_eq!(Stage::Console.back(), None);

        let directory =
            env::temp_dir().join(format!("redrock-stage-{}-{}", std::process::id(), now()));
        fs::create_dir_all(&directory).unwrap();
        assert!(!is_installation(&directory));
        fs::write(directory.join(".redrock-install"), "RedRock installation\n").unwrap();
        assert!(is_installation(&directory));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn installer_writes_runnable_private_layout() {
        let root =
            env::temp_dir().join(format!("redrock-install-{}-{}", std::process::id(), now()));
        let directory = root.join("RedRock body");
        let source = root.join("source");
        let service = root.join("redrock.service");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, "binary").unwrap();

        write_install_files(
            &directory,
            "deep-key",
            "bot-token",
            "42,7",
            &source,
            &service,
        )
        .unwrap();

        assert_eq!(fs::read(directory.join(binary_name())).unwrap(), b"binary");
        assert!(directory.join("skills").is_dir());
        assert!(directory.join("files/inbox").is_dir());
        // The prompt exposes neither source nor a self-update procedure.
        assert!(!directory.join("source").exists());
        assert_eq!(
            installed_version(&directory).as_deref(),
            Some(env!("CARGO_PKG_VERSION"))
        );
        fs::write(directory.join(".redrock-install"), "RedRock installation\n").unwrap();
        assert!(is_installation(&directory) && installed_version(&directory).is_none());
        let stored = fs::read_to_string(directory.join("config.env")).unwrap();
        assert!(stored.contains(&format!(
            "REDROCK_SKILLS={}",
            env_value(&directory.join("skills").to_string_lossy())
        )));
        assert!(stored.contains(&format!(
            "REDROCK_FILES={}",
            env_value(&directory.join("files").to_string_lossy())
        )));
        assert!(stored.contains("REDROCK_ALLOWED_USERS=\"42,7\""));
        assert!(!stored.contains("REDROCK_SOURCE"));
        fs::remove_file(directory.join(".redrock-install")).unwrap();
        assert!(is_installation(&directory));
        fs::remove_file(directory.join(binary_name())).unwrap();
        assert!(!is_installation(&directory));
        assert_eq!(
            parse_allowed_users(&config_value(&directory, "REDROCK_ALLOWED_USERS").unwrap())
                .unwrap(),
            HashSet::from([42, 7])
        );
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(directory.join("config.env"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        // Lifecycle files do not contain authorization data.
        let unit = fs::read_to_string(service).unwrap();
        assert!(!unit.contains("42,7") && !unit.contains("telegram 42"));
        #[cfg(target_os = "linux")]
        {
            assert!(unit.contains(&format!(
                "EnvironmentFile={}",
                directory.join("config.env").display()
            )));
            assert!(unit.contains(&format!("WorkingDirectory={}", directory.display())));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn windows_autostart_command_starts_the_agent() {
        let command = windows_autostart_command(Path::new(r"C:\Users\Me\RedRock\redrock.exe"));

        assert_eq!(command, r#""C:\Users\Me\RedRock\redrock.exe" telegram"#);
    }

    fn message(user_id: i64, text: &str) -> Value {
        json!({"message":{"from":{"id":user_id},"chat":{"type":"private"},"text":text}})
    }

    fn mock(responses: Vec<(&'static str, Value)>) -> (String, thread::JoinHandle<Vec<Value>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut received = Vec::new();
            for (expected, body) in responses {
                let mut stream = listener.accept().unwrap().0;
                let (path, request) = read_request(&stream);
                assert_eq!(path, expected);
                received.push(request);
                reply(&mut stream, body);
            }
            received
        });
        (format!("http://{address}"), server)
    }

    /// Run a queued turn to completion in tests.
    fn run_turn(agent: &Agent, state: &mut State, allowed: &HashSet<i64>) {
        begin_turn(agent, state);
        while state.active.is_some() {
            let (request, read_to) = start_active_step(agent, state, allowed).unwrap();
            finish_active_step(agent, state, run_step(request), read_to);
        }
    }

    fn run_wake_step(agent: &Agent, state: &mut State, allowed: &HashSet<i64>) {
        let request = start_wake_step(agent, state, allowed).unwrap();
        finish_wake_step(state, run_step(request));
    }

    /// Configure all service endpoints for the test server.
    fn test_agent(api: &str) -> Agent {
        Agent {
            client: client().unwrap(),
            key: "key".into(),
            model_api: api.into(),
            token: "test".into(),
            telegram_api: api.into(),
            budget: 500_000,
            files: env::temp_dir().join(format!("redrock-test-files-{}", std::process::id())),
        }
    }

    fn read_request(stream: &TcpStream) -> (String, Value) {
        let (path, _, body) = read_raw_request(stream);
        (path, serde_json::from_slice(&body).unwrap())
    }

    fn read_raw_request(stream: &TcpStream) -> (String, String, Vec<u8>) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let path = line.split_whitespace().nth(1).unwrap().to_owned();
        let mut length = 0;
        let mut headers = String::new();
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        (path, headers, body)
    }

    fn reply(stream: &mut TcpStream, body: Value) {
        let body = body.to_string();
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
    }

    fn reply_bytes(stream: &mut TcpStream, body: &[u8]) {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(body).unwrap();
    }
}
