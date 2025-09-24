#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelKey {
    pub group: String,
    pub model: String,
}

impl ModelKey {
    pub fn new<G: Into<String>, M: Into<String>>(group: G, model: M) -> Self {
        Self {
            group: group.into(),
            model: model.into(),
        }
    }
}
