#![allow(dead_code)]

pub mod list_models {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct ModelPermission {
        id: String,
        object: String,
        created: i64,
        allow_create_engine: bool,
        allow_sampling: bool,
        allow_logprobs: bool,
        allow_search_indices: bool,
        allow_view: bool,
        allow_fine_tuning: bool,
        organization: String,
        group: Option<String>,
        is_blocking: bool,
    }

    #[derive(Debug, Deserialize)]
    pub struct Model {
        id: String,
        object: String,
        created: i64,
        owned_by: String,
        permission: Vec<ModelPermission>,
        root: String,
        parent: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    pub struct ModelList {
        object: String,
        data: Vec<Model>,
    }
}

pub mod edits {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize)]
    pub struct EditParameters {
        model: String,
        input: String,
        instructions: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct EditResponse {
        object: String,
        created: i64,
        choices: Vec<Choice>,
        usage: Usage,
    }

    #[derive(Debug, Deserialize)]
    pub struct Choice {
        text: String,
        index: i32,
    }

    #[derive(Debug, Deserialize)]
    pub struct Usage {
        prompt_tokens: i32,
        completion_tokens: i32,
        total_tokens: i32,
    }
}

pub mod completions {

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize)]
    pub struct CompletionParameters {
        model: String,
        prompt: String,
        max_tokens: i32,
        temperature: f32,
        top_p: f32,
        n: i32,
        stream: bool,
        logprobs: Option<i32>,
        stop: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CompletionResponse {
        id: String,
        object: String,
        created: i64,
        model: String,
        choices: Vec<CompletionChoice>,
        usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CompletionChoice {
        text: String,
        index: i32,
        logprobs: Option<i32>,
        finish_reason: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Usage {
        prompt_tokens: i32,
        completion_tokens: i32,
        total_tokens: i32,
    }
}

pub mod chat {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatParameters {
        model: String,
        messages: Vec<Message>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatResponse {
        id: String,
        object: String,
        created: i64,
        choices: Vec<CompletionChoice>,
        usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CompletionChoice {
        index: i32,
        message: Message,
        finish_reason: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Message {
        role: String,
        content: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Usage {
        prompt_tokens: i32,
        completion_tokens: i32,
        total_tokens: i32,
    }
}
