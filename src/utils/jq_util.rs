use jaq_core::{
    Ctx, RcIter,
    load::{Arena, File, Loader},
};
use jaq_json::Val;
extern crate alloc;
use alloc::vec::Vec;
use indexmap::IndexMap;
use serde_json::Value;
use std::rc::Rc;

pub fn run_jaq(filter: &str, input: &serde_json::Value) -> Option<String> {
    let arena = Arena::default();
    let loader = Loader::new(jaq_std::defs().chain(jaq_json::defs()));
    let modules = loader
        .load(
            &arena,
            File {
                path: (),
                code: filter,
            },
        )
        .unwrap();
    let filter = jaq_core::Compiler::default()
        .with_funs(jaq_std::funs().chain(jaq_json::funs()))
        .compile(modules)
        .unwrap();

    let inputs = RcIter::new(core::iter::empty());
    let ctx = Ctx::new([], &inputs);
    let input_val: Val = json_to_jaq_val(input);
    let mut outputs = filter.run((ctx, input_val));

    if let Some(first_output) = outputs.next() {
        match first_output {
            Ok(val) => Some(jaq_val_to_json_string(&val)),
            Err(_) => None,
        }
    } else {
        None
    }
}

pub fn check_jaq_filter(filter: &str) -> bool {
    let arena = Arena::default();
    let loader = Loader::new(jaq_std::defs().chain(jaq_json::defs()));
    let modules = loader.load(
        &arena,
        File {
            path: (),
            code: filter,
        },
    );
    modules.is_ok()
}

fn json_to_jaq_val(value: &serde_json::Value) -> Val {
    match value {
        Value::Null => Val::Null,
        Value::Bool(b) => Val::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Val::Int(i as isize)
            } else if let Some(f) = n.as_f64() {
                Val::Float(f.into())
            } else {
                Val::Null
            }
        }
        Value::String(s) => Val::Str(s.clone().into()),
        Value::Array(arr) => {
            let jaq_array: Vec<Val> = arr.iter().map(json_to_jaq_val).collect();
            Val::Arr(Rc::new(jaq_array))
        }
        Value::Object(obj) => {
            let mut jaq_object = IndexMap::default();
            for (k, v) in obj {
                jaq_object.insert(Rc::new(k.clone()), json_to_jaq_val(v));
            }
            Val::Obj(Rc::new(jaq_object))
        }
    }
}

fn jaq_val_to_json_string(val: &Val) -> String {
    match val {
        Val::Null => "null".to_string(),
        Val::Bool(b) => b.to_string(),
        Val::Int(i) => i.to_string(),
        Val::Float(f) => {
            if f.is_finite() {
                f.to_string()
            } else {
                "null".to_string()
            }
        }
        Val::Num(n) => n.to_string(),
        Val::Str(s) => s.to_string(),
        Val::Arr(arr) => {
            let elements: Vec<String> = arr.iter().map(jaq_val_to_json_string).collect();
            format!("[{}]", elements.join(","))
        }
        Val::Obj(obj) => {
            let pairs: Vec<String> = obj
                .iter()
                .map(|(k, v)| {
                    let key =
                        serde_json::to_string(k.as_ref()).unwrap_or_else(|_| "\"\"".to_string());
                    format!("{}:{}", key, jaq_val_to_json_string(v))
                })
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_run_jaq() {
        let input = serde_json::json!({
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "What is the weather like in Paris today?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                    },
                    "strict": true
                }
            ]
        });
        assert_eq!(run_jaq("has(\"model\")", &input), Some("true".to_string()));
        assert_eq!(run_jaq(".model", &input), Some("gpt-5".to_string()));
        assert_eq!(
            run_jaq(".model == \"gpt-5\"", &input),
            Some("true".to_string())
        );
        assert_eq!(
            run_jaq(".tools | length > 0", &input),
            Some("true".to_string())
        );
        assert_eq!(
            run_jaq(".tools[].name == \"get_weather\"", &input),
            Some("true".to_string())
        );
    }

    #[tokio::test]
    async fn test_check_jaq_filter() {
        assert!(check_jaq_filter("has(\"model\")"));
        assert!(!check_jaq_filter("has(\"model"));
        assert!(!check_jaq_filter(""));
    }
}
