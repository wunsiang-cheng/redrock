use crate::Result;
use reqwest::blocking::Client;
use std::{collections::HashSet, env, fs, path::Path, time::Duration};

pub(crate) const DEFAULT_MODEL: &str = "deepseek-v4-pro";
pub(crate) const DEEPSEEK_API: &str = "https://api.deepseek.com";
pub(crate) const TELEGRAM_API: &str = "https://api.telegram.org";
const DEFAULT_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_CONTEXT_RATIO: f64 = 0.5;

pub(crate) fn required_env(name: &str) -> Result<String> {
    configured(name).ok_or_else(|| format!("{name} environment variable is required").into())
}

pub(crate) fn configured(name: &str) -> Option<String> {
    if let Ok(value) = env::var(name) {
        return Some(value);
    }
    config_value(env::current_exe().ok()?.parent()?, name)
}

pub(crate) fn config_value(directory: &Path, name: &str) -> Option<String> {
    fs::read_to_string(directory.join("config.env"))
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{name}=")).map(parse_env_value))
}

fn parse_env_value(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
}

pub(crate) fn allowed_users() -> Result<HashSet<i64>> {
    parse_allowed_users(&configured("REDROCK_ALLOWED_USERS").unwrap_or_default())
}

pub(crate) fn parse_allowed_users(value: &str) -> Result<HashSet<i64>> {
    let users = value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            entry.parse::<i64>().map_err(|_| {
                format!("REDROCK_ALLOWED_USERS holds {entry:?}, which is not a Telegram user ID")
            })
        })
        .collect::<std::result::Result<HashSet<i64>, _>>()?;
    if users.is_empty() {
        return Err("REDROCK_ALLOWED_USERS must list at least one Telegram user ID".into());
    }
    Ok(users)
}

pub(crate) fn client() -> Result<Client> {
    Ok(Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?)
}

pub(crate) fn model() -> String {
    configured("REDROCK_MODEL").unwrap_or_else(|| DEFAULT_MODEL.into())
}

pub(crate) fn context_budget() -> Result<usize> {
    let tokens = configured("REDROCK_CONTEXT_TOKENS")
        .map_or(Ok(DEFAULT_CONTEXT_TOKENS), |value| value.parse::<usize>())?;
    if tokens == 0 {
        return Err("REDROCK_CONTEXT_TOKENS must be greater than 0".into());
    }
    Ok((tokens as f64 * context_ratio()?) as usize)
}

fn context_ratio() -> Result<f64> {
    let ratio = configured("REDROCK_CONTEXT_RATIO")
        .map_or(Ok(DEFAULT_CONTEXT_RATIO), |value| value.parse::<f64>())?;
    if !(0.0..=1.0).contains(&ratio) || ratio == 0.0 {
        return Err("REDROCK_CONTEXT_RATIO must be greater than 0 and at most 1".into());
    }
    Ok(ratio)
}
