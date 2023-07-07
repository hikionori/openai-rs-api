#![allow(dead_code)]
use crate::core::{
    models::chat::{ChatParameters, Message, CompletionChoice},
    OpenAI
};

#[derive(Clone)]
pub struct Chat {
    pub oai: OpenAI,
    pub model: ChatModel,
    pub messages: Vec<Message>,
}

pub enum ChatRole {
    System,
    User,
    Assistant,
    Function,
}

#[derive(Clone)]
pub enum ChatModel {
    Gpt4,
    Gpt40613,
    Gpt432k,
    Gpt432k0613,
    Gpt35Turbo,
    Gpt35Turbo0613,
    Gpt35Turbo16k,
    Gpt35Turbo16k0613,
}

impl Default for ChatModel {
    fn default() -> Self {
        ChatModel::Gpt35Turbo
    }
}

impl Into<String> for ChatModel {
    fn into(self) -> String {
        match self {
            ChatModel::Gpt4 => "gpt-4".to_string(),
            ChatModel::Gpt40613 => "gpt-4-0613".to_string(),
            ChatModel::Gpt432k => "gpt-4-32k".to_string(),
            ChatModel::Gpt432k0613 => "gpt-4-32k-0613".to_string(),
            ChatModel::Gpt35Turbo => "gpt-3.5-turbo".to_string(),
            ChatModel::Gpt35Turbo0613 => "gpt-3.5-turbo-0613".to_string(),
            ChatModel::Gpt35Turbo16k => "gpt-3.5-turbo-16k".to_string(),
            ChatModel::Gpt35Turbo16k0613 => "gpt-3.5-turbo-16k-0613".to_string(),
        }
    }
}

impl Into<String> for ChatRole {
    fn into(self) -> String {
        match self {
            ChatRole::System => "system".to_string(),
            ChatRole::User => "user".to_string(),
            ChatRole::Assistant => "assistant".to_string(),
            ChatRole::Function => "function".to_string(),
        }
    }
}

impl Chat {
    fn new(openai_api_key: String, openai_org_key: String, model: ChatModel) -> Self {
        let oai = OpenAI::new(openai_api_key, openai_org_key);
        let messages = Vec::new();
        Self {
            oai,
            messages,
            model,
        }
    }

    fn get_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    async fn send_message(&mut self, message: String, role: ChatRole) -> Vec<CompletionChoice> {
        let message = Message {
            role: role.into(),
            content: message.into(),
            name: None,
            function_call: None,
        };
        self.messages.push(message.clone());

        let parameters = ChatParameters::new(self.model.clone().into(), self.get_messages());

        let response = self
            .oai
            .clone()
            .create_chat_completions(parameters)
            .await
            .unwrap();

        let choices = response.choices;
        
        self.messages.push(
            choices.get(0).unwrap().message.clone()
        );

        choices
    }


}
