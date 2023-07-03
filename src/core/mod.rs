mod models;

use std::{error::Error, sync::Arc};

use awc::Client;

use rustls::{ClientConfig, OwnedTrustAnchor, RootCertStore};

use self::models::{
    chat::{ChatParameters, ChatResponse},
    completions::{CompletionParameters, CompletionResponse},
    edits::{EditParameters, EditResponse},
    list_models::{Model, ModelList},
};

pub struct OpenAI {
    pub token: String,
    pub oia_org: String,
    https_client: Client,
}

impl OpenAI {
    /// The function creates a new instance of the OpenAI struct with the provided token and
    /// organization, along with an HTTPS client.
    /// 
    /// Arguments:
    /// 
    /// * `token`: The `token` parameter is a string that represents the authentication token used to
    /// access the OpenAI API. This token is typically provided by OpenAI when you sign up for their
    /// services.
    /// * `oia_org`: The `oia_org` parameter represents the OpenAI organization ID. It is used to
    /// identify the organization associated with the API token being used.
    /// 
    /// Returns:
    /// 
    /// The `new` function returns an instance of the `OpenAI` struct.
    pub fn new(token: String, oia_org: String) -> Self {
        let https_client = Client::builder()
            .connector(awc::Connector::new().rustls(Arc::new(Self::rustls_config())))
            .finish();
        OpenAI {
            token,
            oia_org,
            https_client,
        }
    }

    fn rustls_config() -> ClientConfig {
        let mut root_store = RootCertStore::empty();
        root_store.add_server_trust_anchors(webpki_roots::TLS_SERVER_ROOTS.0.iter().map(|ta| {
            OwnedTrustAnchor::from_subject_spki_name_constraints(
                ta.subject,
                ta.spki,
                ta.name_constraints,
            )
        }));

        rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(root_store)
            .with_no_client_auth()
    }

    /// The function `list_models` sends a GET request to the OpenAI API to retrieve a list of models
    /// and returns the parsed response as a `ModelList` object.
    /// 
    /// Returns:
    /// 
    /// a Result object with the type ModelList.
    pub async fn list_models(self) -> Result<ModelList, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/models");

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<ModelList>(&result).expect("Failed to parse model list"))
    }

    /// The function retrieves a model from the OpenAI API using an HTTPS client and returns the parsed
    /// model response.
    /// 
    /// Arguments:
    /// 
    /// * `model`: The `model` parameter is a `String` that represents the name of the model you want to
    /// retrieve. It is used to construct the URL for the API request.
    /// 
    /// Returns:
    /// 
    /// a Result object with the type Model as the Ok variant and Box<dyn Error> as the Err variant.
    pub async fn retrive_model(self, model: String) -> Result<Model, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!("https://api.openai.com/v1/models/{}", model);

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<Model>(&result).expect("Failed to parse model response"))
    }

    /// The function `create_chat_completions` sends a POST request to the OpenAI API to generate chat
    /// completions based on the provided parameters.
    /// 
    /// Arguments:
    /// 
    /// * `parameters`: The `parameters` parameter in the `create_chat_completions` is struct
    /// `ChatParameters`. It represents the input data for the chat completion API. The `ChatParameters`
    /// struct contains the following fields:
    /// ```rust
    ///  model: String,
    ///  messages: Vec<Message>,
    /// ```
    /// Returns:
    /// 
    /// a `Result` type with the `Ok` variant containing a `ChatResponse` object if the operation is
    /// successful, or the `Err` variant containing a `Box<dyn Error>` if an error occurs.
    pub async fn create_chat_completions(
        self,
        parameters: ChatParameters,
    ) -> Result<ChatResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/chat/completions");

        let result = client
            .post(url)
            .insert_header(("Content-Type", "application/json"))
            .bearer_auth(self.token)
            .send_json(&parameters)
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<ChatResponse>(&result).expect("Failed to parse chat response"))
    }

    /// The function `create_completions` sends a POST request to the OpenAI API to generate completions
    /// based on the given parameters.
    /// 
    /// Arguments:
    /// 
    /// * `parameters`: The `parameters` parameter in the `create_completions` function is of type
    /// `CompletionParameters`. It is an input parameter that contains the information required to
    /// generate completions using the OpenAI API.
    /// 
    /// Returns:
    /// 
    /// a `Result` with a `CompletionResponse` on success or a `Box<dyn Error>` on failure.
    pub async fn create_completions(
        self,
        parameters: CompletionParameters,
    ) -> Result<CompletionResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/completions");

        let result = client
            .post(url)
            .insert_header(("Content-Type", "application/json"))
            .bearer_auth(self.token)
            .send_json(&parameters)
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<CompletionResponse>(&result)
            .expect("Failed to parse completion response"))
    }

    /// The function `create_edit` sends a POST request to the OpenAI API to create or edit a completion
    /// and returns the response.
    /// 
    /// Arguments:
    /// 
    /// * `parameters`: The `parameters` parameter in the `create_edit` function is of type
    /// `EditParameters`. It is an input parameter that contains the necessary information for creating
    /// an edit. The specific fields and their meanings depend on the implementation of the
    /// `EditParameters` struct. You would need to refer to the definition
    /// 
    /// Returns:
    /// 
    /// a `Result` with the type `EditResponse` on success or a `Box<dyn Error>` on failure.
    pub async fn create_edit(
        self,
        parameters: EditParameters,
    ) -> Result<EditResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/completions");

        let result = client
            .post(url)
            .insert_header(("Content-Type", "application/json"))
            .bearer_auth(self.token)
            .send_json(&parameters)
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<EditResponse>(&result).expect("Failed to parse edit response"))
    }
}
