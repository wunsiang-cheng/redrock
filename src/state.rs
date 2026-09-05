use crate::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Continuity {
    pub(crate) current_goal: String,
    pub(crate) long_term_memory: String,
    pub(crate) next_wake: i64,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct State {
    pub(crate) histories: HashMap<i64, Vec<Value>>,
    pub(crate) shared_history: Vec<Value>,
    pub(crate) waking: bool,
    pub(crate) active: Option<Active>,
    pub(crate) queue: Vec<i64>,
    pub(crate) continuity: Continuity,
    pub(crate) telegram_offset: i64,
    pub(crate) pending_files: Vec<PendingFile>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct PendingFile {
    pub(crate) update_id: i64,
    pub(crate) user_id: i64,
    pub(crate) file_id: String,
    pub(crate) file_unique_id: String,
    pub(crate) file_name: String,
    pub(crate) mime_type: Option<String>,
    pub(crate) file_size: Option<u64>,
    pub(crate) caption: String,
}

/// The active user turn, advanced one model call at a time.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Active {
    pub(crate) user: i64,
    /// The active Telegram progress message.
    pub(crate) progress: Option<i64>,
    pub(crate) started: i64,
    pub(crate) steps: u32,
    pub(crate) spoke: bool,
}

/// Lock the state file for the process lifetime.
pub(crate) fn open_database(path: &Path) -> Result<Connection> {
    let database = Connection::open(path)?;
    database.pragma_update(None, "locking_mode", "exclusive")?;
    database
        .execute_batch("BEGIN EXCLUSIVE; COMMIT;")
        .map_err(|error| {
            format!(
                "another RedRock is already running on {}: {error}",
                path.display()
            )
        })?;
    database.execute(
        "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL)",
        [],
    )?;
    Ok(database)
}

pub(crate) fn load_state(database: &Connection) -> Result<State> {
    let json: Option<String> = database
        .query_row("SELECT json FROM state WHERE id = 1", [], |row| row.get(0))
        .optional()?;
    Ok(json.map_or_else(|| Ok(State::default()), |json| serde_json::from_str(&json))?)
}

pub(crate) fn save_state(database: &Connection, state: &State) -> Result<()> {
    database.execute(
        "INSERT INTO state (id, json) VALUES (1, ?1) ON CONFLICT(id) DO UPDATE SET json = excluded.json",
        params![serde_json::to_string(state)?],
    )?;
    Ok(())
}
