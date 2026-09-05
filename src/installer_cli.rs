use crate::{
    DISCLOSURE, Result, capture_contact, default_install_directory, install, is_installation,
    parse_allowed_users, run_installer,
};
use std::{
    env,
    io::{IsTerminal, Write},
    path::PathBuf,
};

#[cfg(target_os = "linux")]
use std::process::Command;

#[derive(Default, Debug, PartialEq)]
struct InstallOptions {
    cli: bool,
    gui: bool,
    non_interactive: bool,
    accept_risk: bool,
    directory: Option<PathBuf>,
}

pub(crate) fn run_install_command(arguments: Vec<String>) -> Result<()> {
    let options = parse_options(arguments)?;
    if options.gui || (!options.cli && graphical_session()) {
        run_installer()
    } else {
        run_cli(options)
    }
}

fn parse_options(arguments: Vec<String>) -> Result<InstallOptions> {
    let mut options = InstallOptions::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cli" => options.cli = true,
            "--gui" => options.gui = true,
            "--non-interactive" => {
                options.cli = true;
                options.non_interactive = true;
            }
            "--accept-risk" => {
                options.cli = true;
                options.accept_risk = true;
            }
            "--directory" => {
                options.cli = true;
                let path = arguments.next().ok_or("--directory requires a path")?;
                if path.is_empty() {
                    return Err("--directory requires a path".into());
                }
                options.directory = Some(PathBuf::from(path));
            }
            _ => return Err(format!("unknown install option: {argument}").into()),
        }
    }
    if options.cli && options.gui {
        return Err("--cli and --gui cannot be used together".into());
    }
    if options.gui
        && (options.non_interactive || options.accept_risk || options.directory.is_some())
    {
        return Err("CLI install options cannot be used with --gui".into());
    }
    Ok(options)
}

fn run_cli(options: InstallOptions) -> Result<()> {
    if !options.non_interactive && !std::io::stdin().is_terminal() {
        return Err(
            "interactive CLI installation requires a terminal; use --non-interactive with DEEPSEEK_API_KEY, TELEGRAM_BOT_TOKEN, REDROCK_ALLOWED_USERS, and --accept-risk"
                .into(),
        );
    }
    if options.non_interactive && !options.accept_risk {
        return Err("--non-interactive requires --accept-risk".into());
    }
    if options.non_interactive {
        let missing = [
            "DEEPSEEK_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "REDROCK_ALLOWED_USERS",
        ]
        .into_iter()
        .filter(|name| process_env(name).is_none())
        .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "--non-interactive requires these environment variables: {}",
                missing.join(", ")
            )
            .into());
        }
    }

    println!("RedRock CLI installation");
    if !options.accept_risk {
        println!("\n{DISCLOSURE}\n");
        if prompt("Type 'yes' to accept and continue", None)? != "yes" {
            return Err("installation cancelled".into());
        }
    }

    let default_directory = default_install_directory();
    let directory = match options.directory.as_ref() {
        Some(directory) => directory.clone(),
        None if options.non_interactive => default_directory,
        None => PathBuf::from(prompt(
            "Installation directory",
            Some(&default_directory.to_string_lossy()),
        )?),
    };
    let key = credential("DEEPSEEK_API_KEY", "DeepSeek API key: ", &options)?;
    let token = credential("TELEGRAM_BOT_TOKEN", "Telegram bot token: ", &options)?;
    let users = allowed_users(&token, &options)?;
    parse_allowed_users(&users)?;

    println!("\nInstallation directory: {}", directory.display());
    println!("Allowed Telegram users: {users}");
    if is_installation(&directory) {
        println!("The existing RedRock installation will be updated and reconfigured.");
    }
    if !options.non_interactive && prompt("Install and start RedRock?", Some("no"))? != "yes" {
        return Err("installation cancelled".into());
    }

    let was_installed = is_installation(&directory);
    println!("Verifying credentials, installing files, and starting RedRock…");
    if let Err(error) = install(&directory.to_string_lossy(), &key, &token, &users) {
        if !was_installed && is_installation(&directory) {
            return Err(format!(
                "files were installed at {}, but the service did not start: {error}",
                directory.display()
            )
            .into());
        }
        return Err(error);
    }
    println!("Installed and started RedRock at {}.", directory.display());
    print_linger_notice();
    Ok(())
}

fn allowed_users(token: &str, options: &InstallOptions) -> Result<String> {
    if let Some(users) = process_env("REDROCK_ALLOWED_USERS") {
        return Ok(users);
    }
    if options.non_interactive {
        return Err("REDROCK_ALLOWED_USERS is required with --non-interactive".into());
    }
    let listed = prompt(
        "Allowed Telegram user IDs, separated by commas (leave blank to detect)",
        None,
    )?;
    if !listed.is_empty() {
        return Ok(listed);
    }
    println!("Send a private message to the bot. Waiting for up to two minutes…");
    let step = capture_contact(token, "")?;
    println!("{}", step.message);
    step.users
        .ok_or_else(|| "Telegram user detection returned no ID".into())
}

fn credential(name: &str, label: &str, options: &InstallOptions) -> Result<String> {
    if let Some(value) = process_env(name) {
        return Ok(value);
    }
    if options.non_interactive {
        return Err(format!("{name} is required with --non-interactive").into());
    }
    let value = rpassword::prompt_password(label)?;
    if value.is_empty() {
        return Err(format!("{name} must not be empty").into());
    }
    Ok(value)
}

fn process_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn prompt(label: &str, default: Option<&str>) -> Result<String> {
    match default {
        Some(default) => print!("{label} [{default}]: "),
        None => print!("{label}: "),
    }
    std::io::stdout().flush()?;
    let mut value = String::new();
    std::io::stdin().read_line(&mut value)?;
    let value = value.trim().to_owned();
    Ok(if value.is_empty() {
        default.unwrap_or_default().to_owned()
    } else {
        value
    })
}

#[cfg(target_os = "linux")]
fn graphical_session() -> bool {
    ["DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"]
        .iter()
        .any(|name| env::var(name).is_ok_and(|value| !value.is_empty()))
}

#[cfg(not(target_os = "linux"))]
fn graphical_session() -> bool {
    true
}

#[cfg(target_os = "linux")]
fn print_linger_notice() {
    let Ok(identity) = Command::new("id").arg("-un").output() else {
        return;
    };
    let user = String::from_utf8_lossy(&identity.stdout).trim().to_owned();
    if !identity.status.success() || user.is_empty() {
        return;
    }
    let Ok(output) = Command::new("loginctl")
        .args(["show-user", &user, "-p", "Linger", "--value"])
        .output()
    else {
        return;
    };
    if output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == "no" {
        println!(
            "The service may stop after logout. To keep it running, run: sudo loginctl enable-linger {user}"
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn print_linger_notice() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_select_cli_without_accepting_secret_arguments() {
        let options = parse_options(vec![
            "--cli".into(),
            "--non-interactive".into(),
            "--accept-risk".into(),
            "--directory".into(),
            "/srv/redrock".into(),
        ])
        .unwrap();
        assert_eq!(
            options,
            InstallOptions {
                cli: true,
                gui: false,
                non_interactive: true,
                accept_risk: true,
                directory: Some(PathBuf::from("/srv/redrock")),
            }
        );
        assert!(parse_options(vec!["--cli".into(), "--gui".into()]).is_err());
        assert!(parse_options(vec!["--gui".into(), "--accept-risk".into()]).is_err());
        assert!(parse_options(vec!["--telegram-token".into(), "secret".into()]).is_err());
    }
}
