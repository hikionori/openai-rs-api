# OpenAI Rust Api
Unofficial api client for OpenAI

**[Functions of `core` module](./core.md)**

#### Example of usage `core` module

### Chat completion
```rust
use openai_rs_api::core::models::chat::{ChatParameters, ChatResponse, Message};
use openai_rs_api::core::OpenAI;

#[tokio::main]
async fn main() {
    let oai = OpenAI::new("API_KEY".to_string(), "API_ORG".to_string());
    let parameters = ChatParameters::new(
        "gpt-3.5-turbo-16k".to_string(),
        vec![Message {
            role: "user".to_string(),
            content: Some("Hello, my name is".to_string()),
            name: None,
            function_call: None,
        }],
    );
    let response: ChatResponse = oai.create_chat_completions(parameters).await.unwrap();
    // print answer message
    if let Some(content) = &response.choices.get(0).unwrap().message.content {
        println!("{}", content);
    }
}
```