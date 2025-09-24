use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use super::types::ModelKey;
use crate::config::{Config, ModelGroupEntry};

pub struct Health {
    // factor in percentage points (100 = 1.0x)
    factors: HashMap<ModelKey, AtomicU32>,
    breaker: Mutex<HashMap<ModelKey, Breaker>>, // protected as it carries Instants
    cfg: HealthConfig,
}

impl Health {
    pub fn new_from_config(cfg: &Config) -> Self {
        let mut factors = HashMap::new();
        let mut breaker = HashMap::new();
        for g in &cfg.router_settings.model_groups {
            for m in &g.models {
                let key = ModelKey::new(g.name.clone(), m.name.clone());
                factors.insert(key.clone(), AtomicU32::new(100));
                breaker.insert(key, Breaker::default());
            }
        }
        Self {
            factors,
            breaker: Mutex::new(breaker),
            cfg: HealthConfig::default(),
        }
    }

    pub fn effective_weight(&self, group_name: &str, entry: &ModelGroupEntry) -> u32 {
        let key = ModelKey::new(group_name.to_string(), entry.name.clone());
        let base = entry.weight;
        let factor = self
            .factors
            .get(&key)
            .map(|a| a.load(Ordering::SeqCst))
            .unwrap_or(100);
        let mut eff = (base as u64 * factor as u64) / 100;
        if base > 0 && eff == 0 {
            eff = 1;
        }
        eff as u32
    }

    pub fn decay(&self, key: &ModelKey) {
        if let Some(f) = self.factors.get(key) {
            loop {
                let cur = f.load(Ordering::SeqCst);
                let next = (cur / 2).max(1);
                if f.compare_exchange_weak(cur, next, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    break;
                }
            }
        }
    }

    pub fn recover_on_success(&self, key: &ModelKey) {
        if let Some(f) = self.factors.get(key) {
            loop {
                let cur = f.load(Ordering::SeqCst);
                if cur >= 100 {
                    break;
                }
                let step = self.cfg.recovery_step;
                let mut next = cur.saturating_add(step);
                if next > 100 {
                    next = 100;
                }
                if f.compare_exchange_weak(cur, next, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    break;
                }
            }
        }
        // Close/half-open transitions
        let mut map = self.breaker.lock().unwrap();
        if let Some(b) = map.get_mut(key) {
            b.consecutive_failures = 0;
            if let CircuitState::HalfOpen = b.state {
                b.state = CircuitState::Closed;
                b.open_until = None;
            }
        }
    }

    pub fn on_failure(&self, key: &ModelKey) {
        let mut map = self.breaker.lock().unwrap();
        let b = map.entry(key.clone()).or_insert_with(Breaker::default);
        b.consecutive_failures = b.consecutive_failures.saturating_add(1);
        if b.consecutive_failures >= self.cfg.fail_threshold {
            b.state = CircuitState::Open;
            b.open_until = Some(Instant::now() + self.cfg.open_duration);
        }
    }

    pub fn permit(&self, group_name: &str, entry: &ModelGroupEntry) -> bool {
        let key = ModelKey::new(group_name.to_string(), entry.name.clone());
        let mut map = self.breaker.lock().unwrap();
        let b = map.entry(key).or_insert_with(Breaker::default);
        match b.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true, // allow probing
            CircuitState::Open => {
                if let Some(t) = b.open_until {
                    if Instant::now() >= t {
                        b.state = CircuitState::HalfOpen;
                        b.open_until = None;
                        true
                    } else {
                        false
                    }
                } else {
                    // Safety: if open but no deadline, allow after default duration
                    b.state = CircuitState::HalfOpen;
                    true
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Clone, Debug)]
struct Breaker {
    state: CircuitState,
    consecutive_failures: u32,
    open_until: Option<Instant>,
}

impl Default for Breaker {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            open_until: None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct HealthConfig {
    pub fail_threshold: u32,
    pub open_duration: Duration,
    pub recovery_step: u32, // percentage points per success
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            fail_threshold: 3,
            open_duration: Duration::from_secs(30),
            recovery_step: 10,
        }
    }
}
