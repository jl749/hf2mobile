//! Wrap a prompt the way the model was instruction-tuned to expect.
//!
//! `hf2mobile.infer` hands this job to `transformers`
//! (`tokenizer.apply_chat_template(..., add_generation_prompt=True)`). The binary has no
//! interpreter to hand it to, so it renders the same template itself: the `chat_template` in
//! `tokenizer_config.json` is a Jinja2 document, and [`minijinja`] is a Jinja2 engine.
//!
//! Only the single-user-turn case is built here, because that is what the CLI takes — one
//! `--prompt`. The context a template is rendered against is otherwise the same one
//! `transformers` builds: the message list, `add_generation_prompt`, and every `*_token` the
//! tokenizer config names (`bos_token`, `eos_token`, …), which templates splice in by name.
//!
//! # Where the template lives
//!
//! Two places, and an export can only be read by checking both. `transformers` v5 saves the
//! template as its own file, `chat_template.jinja`, next to the tokenizer — that is what the
//! exports in this repo carry. Before that it was a `chat_template` string (or a list of named
//! ones) inside `tokenizer_config.json`, which is still what a model downloaded from an older
//! snapshot has.
//!
//! A model with no template at all — base models, and the byte-level fixture the tests use — is
//! not an error: [`ChatTemplate::load`] returns `None` and the caller sends the prompt through
//! as-is, matching what `hf2mobile.infer` does.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use minijinja::{Environment, Error as JinjaError, ErrorKind as JinjaErrorKind, Value as JinjaValue};
use serde_json::Value as Json;

/// `transformers` >= 5 writes the template here, beside `tokenizer_config.json`.
const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// Where the tokenizer's special tokens live, and older exports' templates with them.
const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";

/// The name of the template inside a `tokenizer_config.json` template *list*; the one
/// `transformers` picks when the caller names no other.
const DEFAULT_TEMPLATE: &str = "default";

/// One model's chat template, plus the special tokens it is allowed to refer to.
pub struct ChatTemplate {
    /// The Jinja source, straight out of `tokenizer_config.json`.
    source: String,
    /// `("bos_token", "<s>")`-shaped pairs — every `*_token` the config spells out as a string.
    /// A `Vec` rather than a map because there are three or four of them and they are only
    /// ever iterated.
    special_tokens: Vec<(String, String)>,
}

impl ChatTemplate {
    /// Read an export directory's chat template, or `None` if it carries none.
    ///
    /// `chat_template.jinja` wins over the `chat_template` key when a directory somehow has
    /// both, because that is the file `transformers` itself would load — an export written by a
    /// newer version, read next to a config left over from an older one.
    pub fn load(export_dir: &Path) -> Result<Option<Self>> {
        let config = read_json(&export_dir.join(TOKENIZER_CONFIG_FILE))?;
        let source = match std::fs::read_to_string(export_dir.join(CHAT_TEMPLATE_FILE)) {
            Ok(jinja) => Some(jinja),
            // `as_ref` borrows the inside of the `Option` so `config` survives to be used again
            // below; `and_then` then calls `template_source` on it, passed here by name rather
            // than wrapped in a closure because its signature already lines up.
            Err(_) => config.as_ref().and_then(template_source),
        };

        // `let ... else` is destructuring that is allowed to fail: it binds `source` for the
        // rest of the function, and the `else` block runs when there was nothing to bind — so
        // it has to leave the function. It is `if let` turned inside out, and it keeps the
        // interesting path un-indented.
        let Some(source) = source else {
            return Ok(None);
        };
        Ok(Some(Self {
            source,
            special_tokens: config.as_ref().map(special_tokens).unwrap_or_default(),
        }))
    }

    /// Render the template for a single user turn, ending where the model should start writing.
    pub fn render(&self, prompt: &str) -> Result<String> {
        let mut env = Environment::new();
        // These templates are written against Jinja2 running on Python, so they call Python's
        // own methods on the values they are handed — Qwen3 does `content.startswith(...)`,
        // others use `.items()` or `.strip()`. Jinja has no such methods and minijinja reports
        // them as "unknown method"; this callback implements the Python ones.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Templates written for `transformers` assume its two helpers exist. `raise_exception`
        // is how a template rejects a conversation it cannot represent (a system message where
        // none is allowed, say), and `strftime_now` dates the system prompt in the Llama 3.x
        // family. Neither is part of Jinja itself, so both are supplied here.
        env.add_function("raise_exception", |message: String| -> Result<JinjaValue, JinjaError> {
            Err(JinjaError::new(JinjaErrorKind::InvalidOperation, message))
        });
        env.add_function("strftime_now", |format: String| strftime_utc(&format));
        env.add_template("chat", &self.source)
            .context("the model's `chat_template` is not valid Jinja")?;

        let mut context = std::collections::BTreeMap::<String, JinjaValue>::new();
        for (name, value) in &self.special_tokens {
            context.insert(name.clone(), JinjaValue::from(value.as_str()));
        }
        // The one message, shaped the way `transformers` shapes it. `content` is a plain
        // string: the multimodal spelling (a list of `{"type": "text", ...}` parts) is what a
        // template falls back to only when it is handed one, and the CLI takes text.
        context.insert(
            "messages".into(),
            JinjaValue::from_serialize(vec![std::collections::BTreeMap::from([
                ("role", "user"),
                ("content", prompt),
            ])]),
        );
        // "…and now it is the assistant's turn": the flag that makes a template emit the
        // opening of a reply rather than stopping after the user's message.
        context.insert("add_generation_prompt".into(), JinjaValue::from(true));

        env.get_template("chat")
            .expect("the template was just added")
            .render(context)
            .context("rendering the model's `chat_template`")
    }
}

/// Parse a JSON file that is allowed not to exist.
///
/// A directory with no `tokenizer_config.json` still has a template if it has the `.jinja`
/// file, so a missing file is `None` rather than an error — but a file that *is* there and does
/// not parse is a broken export and says so.
fn read_json(path: &Path) -> Result<Option<Json>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(None);
    };
    serde_json::from_str(&text)
        .map(Some)
        .with_context(|| format!("`{}` is not valid JSON", path.display()))
}

/// The Jinja source in a parsed `tokenizer_config.json`, in the two shapes it comes in.
///
/// A plain string is the common one. The list of `{"name": ..., "template": ...}` objects is
/// what models that ship a second, tool-calling template use; `transformers` takes the entry
/// called `default` when no name is asked for, and so do we.
fn template_source(config: &Json) -> Option<String> {
    match config.get("chat_template")? {
        Json::String(source) => Some(source.clone()),
        Json::Array(entries) => {
            let named = |wanted: &str| {
                entries
                    .iter()
                    .find(|entry| entry.get("name").and_then(Json::as_str) == Some(wanted))
            };
            let entry = named(DEFAULT_TEMPLATE).or_else(|| entries.first())?;
            entry.get("template").and_then(Json::as_str).map(str::to_owned)
        }
        _ => None,
    }
}

/// Every `*_token` the config gives a string value, as `("bos_token", "<s>")` pairs.
///
/// Collected by suffix rather than from a fixed list because which ones a template reaches for
/// varies by model — Gemma uses `bos_token`, Llama's tool template also wants `eos_token`, and
/// a template that mentions one we did not pass would render the word `undefined` into the
/// prompt. The value is either the token itself or an `AddedToken` object carrying it under
/// `content`, which is how `tokenizers` serializes one it has metadata for.
fn special_tokens(config: &Json) -> Vec<(String, String)> {
    let Some(map) = config.as_object() else {
        return Vec::new();
    };
    map.iter()
        .filter(|(name, _)| name.ends_with("_token"))
        // `filter_map` is a filter and a map in one pass: returning `None` drops the entry,
        // returning `Some(x)` keeps `x`. Used here because deciding whether to keep a key and
        // working out its value are the same question — a `*_token` we cannot read is not one.
        .filter_map(|(name, value)| {
            let token = match value {
                Json::String(token) => Some(token.clone()),
                Json::Object(added) => added.get("content").and_then(Json::as_str).map(str::to_owned),
                _ => None,
            };
            token.map(|token| (name.clone(), token))
        })
        .collect()
}

/// `strftime`, today, in UTC — enough of it for the templates that call it.
///
/// The supported specifiers are the ones the HF templates actually use: `%d %b %Y` (Llama 3.x)
/// and the obvious neighbours. Anything else is left in the string untouched rather than
/// failing the render, since a stray specifier in a system prompt is a cosmetic problem and an
/// error here would cost the user their whole generation.
fn strftime_utc(format: &str) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let (days, seconds_today) = ((secs / 86_400) as i64, secs % 86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (seconds_today / 3600, (seconds_today / 60) % 60, seconds_today % 60);

    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let month_name = MONTHS[(month - 1) as usize];

    format
        .replace("%Y", &year.to_string())
        .replace("%m", &format!("{month:02}"))
        .replace("%d", &format!("{day:02}"))
        .replace("%B", month_name)
        .replace("%b", &month_name[..3])
        .replace("%H", &format!("{hour:02}"))
        .replace("%M", &format!("{minute:02}"))
        .replace("%S", &format!("{second:02}"))
}

/// Days since 1970-01-01 -> `(year, month, day)`, proleptic Gregorian.
///
/// Howard Hinnant's `civil_from_days`, the algorithm every date library uses: it shifts the
/// epoch to March 1st so that the leap day lands at the end of the year, which removes the
/// special-casing February would otherwise need. Written out rather than pulled in with a
/// crate because this is the only date arithmetic in the repo.
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468; // shift the epoch to 0000-03-01
    let era = z.div_euclid(146_097); // one era is 400 years, exactly 146097 days
    let doe = z.rem_euclid(146_097); // day of era, 0..=146096
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // year of era, 0..=399
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year, March-based
    let mp = (5 * doy + 2) / 153; // month, March-based: 0..=11
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_user_turn_with_the_special_tokens() {
        let config = serde_json::json!({
            "bos_token": "<bos>",
            "eos_token": {"content": "<eos>"},
            "chat_template": "{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}\
                              {% if add_generation_prompt %}<model>{% endif %}",
        });
        let template = ChatTemplate {
            source: template_source(&config).unwrap(),
            special_tokens: special_tokens(&config),
        };
        assert_eq!(template.render("hi").unwrap(), "<bos><user>hi<model>");
    }

    #[test]
    fn a_config_with_no_template_is_not_an_error() {
        assert!(template_source(&serde_json::json!({"bos_token": "<bos>"})).is_none());
    }

    /// The two on-disk layouts, and which one wins.
    #[test]
    fn load_reads_either_layout() {
        let dir = std::env::temp_dir().join("hf2mobile-chat-template-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join(TOKENIZER_CONFIG_FILE);
        let jinja = dir.join(CHAT_TEMPLATE_FILE);

        // Nothing at all: a base model, not an error.
        assert!(ChatTemplate::load(&dir).unwrap().is_none());

        // The pre-v5 layout: the template inside the tokenizer config.
        std::fs::write(
            &config,
            r#"{"bos_token": "<s>", "chat_template": "{{ bos_token }}old"}"#,
        )
        .unwrap();
        assert_eq!(
            ChatTemplate::load(&dir).unwrap().unwrap().render("hi").unwrap(),
            "<s>old"
        );

        // The v5 layout: its own file, and it wins — `transformers` would load it too. The
        // special tokens still come from the config beside it.
        std::fs::write(&jinja, "{{ bos_token }}new").unwrap();
        assert_eq!(
            ChatTemplate::load(&dir).unwrap().unwrap().render("hi").unwrap(),
            "<s>new"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn picks_the_default_entry_out_of_a_template_list() {
        let config = serde_json::json!({
            "chat_template": [
                {"name": "tool_use", "template": "tools"},
                {"name": "default", "template": "plain"},
            ]
        });
        assert_eq!(template_source(&config).as_deref(), Some("plain"));
    }

    /// Qwen3's template opens with `messages[0].content.startswith(...)`, which is a Python
    /// method rather than a Jinja one — the whole reason `minijinja-contrib` is a dependency.
    #[test]
    fn python_string_methods_work() {
        let template = ChatTemplate {
            source: "{% if messages[0]['content'].startswith('Where') %}yes{% else %}no{% endif %}".into(),
            special_tokens: Vec::new(),
        };
        assert_eq!(template.render("Where is Paris?").unwrap(), "yes");
    }

    #[test]
    fn raise_exception_fails_the_render_rather_than_the_process() {
        let template = ChatTemplate {
            source: "{{ raise_exception('no system message allowed') }}".into(),
            special_tokens: Vec::new(),
        };
        let err = format!("{:#}", template.render("hi").unwrap_err());
        assert!(err.contains("no system message allowed"), "{err}");
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(59), (1970, 3, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // a leap year, past a century boundary
        assert_eq!(strftime_utc("%d %b %Y").len(), "01 Jan 1970".len());
    }
}
