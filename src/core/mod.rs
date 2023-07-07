pub mod models;

use std::{error::Error, sync::Arc};

use awc::Client;

use rustls::{ClientConfig, OwnedTrustAnchor, RootCertStore};

use self::models::{
    audio::{TextResponse, TranscriptionParameters, TranslationParameters},
    chat::{ChatParameters, ChatResponse},
    completions::{CompletionParameters, CompletionResponse},
    edits::{EditParameters, EditResponse},
    embeddings::{EmbeddingParameters, EmbeddingResponse},
    files::{DeleteResponse, FileData, FileList, FileUpload},
    fine_tunes::{
        CreateFineTuneParameters, FineTuneDelete, FineTuneEventList, FineTuneList,
        FineTuneRetriveData,
    },
    images::{ImageCreateParameters, ImageEditParameters, ImageResponse, ImageVariationParameters},
    list_models::{Model, ModelList},
    moderations::{TextModerationParameters, TextModerationResult},
};

#[derive(Clone)]
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
    #[cfg(feature = "list_models")]
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
    #[cfg(feature = "list_models")]
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

    /// The function `create_chat_completions` sends a POST request to the OpenAI API to generate chat completions
    /// based on the given parameters.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_chat_completions` function is of type
    /// `ChatParameters`. It is an input parameter that contains the information required to
    /// generate chat completions using the OpenAI API.
    ///
    /// Returns:
    ///
    /// a `Result` with a `ChatResponse` on success or a `Box<dyn Error>` on failure.
    #[cfg(feature = "chat")]
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
    #[cfg(feature = "completions")]
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
    #[cfg(feature = "edits")]
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

    /// The `create_image` function sends a POST request to the OpenAI API to generate an image based on
    /// the provided parameters.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_image` function is of type
    /// `ImageCreateParameters`. It is an input parameter that contains the necessary information for
    /// generating an image.
    ///
    /// Returns:
    ///
    /// The function `create_image` returns a `Result` enum with the success case containing an
    /// `ImageResponse` and the error case containing a `Box<dyn Error>`.
    #[cfg(feature = "images")]
    pub async fn create_image(
        self,
        parameters: ImageCreateParameters,
    ) -> Result<ImageResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/images/generations");

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

        Ok(serde_json::from_slice::<ImageResponse>(&result)
            .expect("Failed to parse image response"))
    }

    /// The function `create_image_edit` sends a POST request to the OpenAI API to create an image edit,
    /// using the provided parameters, and returns the resulting image response.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_image_edit` function is of type
    /// `ImageEditParameters`. It is an input parameter that contains the necessary information for
    /// creating an image edit.
    ///
    /// Returns:
    ///
    /// a Result type with the success variant containing an ImageResponse or the error variant
    /// containing a Box<dyn Error>.
    #[cfg(feature = "images")]
    pub async fn create_image_edit(
        self,
        parameters: ImageEditParameters,
    ) -> Result<ImageResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/images/edits");

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

        Ok(serde_json::from_slice::<ImageResponse>(&result)
            .expect("Failed to parse image response"))
    }

    /// The function `create_image_variations` sends a POST request to the OpenAI API to create image
    /// variations based on the provided parameters.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_image_variations` function is of type
    /// `ImageVariationParameters`. It is an input parameter that contains the necessary information for
    /// creating image variations.
    ///
    /// Returns:
    ///
    /// a Result object with the type ImageResponse.
    #[cfg(feature = "images")]
    pub async fn create_image_variations(
        self,
        parameters: ImageVariationParameters,
    ) -> Result<ImageResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/images/variations");

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

        Ok(serde_json::from_slice::<ImageResponse>(&result)
            .expect("Failed to parse image response"))
    }

    /// The function `create_embedding` sends a POST request to the OpenAI API to create an embedding,
    /// using the provided parameters, and returns the resulting embedding response.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_embedding` function is of type
    /// `EmbeddingParameters`. It is an input parameter that contains the necessary information for
    /// creating an embedding.
    ///
    /// Returns:
    ///
    /// a Result type with the success variant containing an EmbeddingResponse or the error variant
    /// containing a Box<dyn Error>.
    #[cfg(feature = "embeddings")]
    pub async fn create_embedding(
        self,
        parameters: EmbeddingParameters,
    ) -> Result<EmbeddingResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/embeddings");

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

        Ok(serde_json::from_slice::<EmbeddingResponse>(&result)
            .expect("Failed to parse embedding response"))
    }

    /// The function `create_transcription` sends a POST request to the OpenAI API to create a transcription,
    /// using the provided parameters, and returns the resulting transcription response.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_transcription` function is of type
    /// `TranscriptionParameters`. It is an input parameter that contains the necessary information for
    /// creating a transcription.
    ///
    /// Returns:
    ///
    /// a Result type with the success variant containing a TextResponse or the error variant
    /// containing a Box<dyn Error>.
    #[cfg(feature = "audio")]
    pub async fn create_transcription(
        self,
        parameters: TranscriptionParameters,
    ) -> Result<TextResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/audio/transcriptions");

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

        Ok(serde_json::from_slice::<TextResponse>(&result).expect("Failed to parse text response"))
    }

    /// The function `create_translation` sends a POST request to the OpenAI API to create a translation,
    /// using the provided parameters, and returns the resulting translation response.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_translation` function is of type
    /// `TranslationParameters`. It is an input parameter that contains the necessary information for
    /// creating a translation.
    ///
    /// Returns:
    ///
    /// a Result type with the success variant containing a TextResponse or the error variant
    /// containing a Box<dyn Error>.
    #[cfg(feature = "audio")]
    pub async fn create_translation(
        self,
        parameters: TranslationParameters,
    ) -> Result<TextResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/audio/translations");

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

        Ok(serde_json::from_slice::<TextResponse>(&result).expect("Failed to parse text response"))
    }

    /// The function `list_files` makes an asynchronous HTTP GET request to the OpenAI API to retrieve a
    /// list of files and returns the parsed result.
    ///
    /// Returns:
    ///
    /// The function `list_files` returns a `Result` containing either a `FileList` or a boxed dynamic
    /// error (`Box<dyn Error>`).
    #[cfg(feature = "files")]
    pub async fn list_files(self) -> Result<FileList, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/files");

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FileList>(&result).expect("Failed to parse file list"))
    }

    /// The `upload_files` function in Rust uploads files to the OpenAI API and returns the file data.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `upload_files` function is of type
    /// `FileUpload`. It represents the data that needs to be uploaded to the server. The `FileUpload`
    /// struct should contain the necessary information for the file upload, such as the file content,
    /// file name, and file type
    ///
    /// Returns:
    ///
    /// The function `upload_files` returns a `Result` containing either a `FileData` object or an error
    /// (`Box<dyn Error>`).
    #[cfg(feature = "files")]
    pub async fn upload_files(self, parameters: FileUpload) -> Result<FileData, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/files");

        let result = client
            .post(url)
            .bearer_auth(self.token)
            .send_json(&parameters)
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FileData>(&result).expect("Failed to parse file data"))
    }

    /// The function `delete_file` is an asynchronous function in Rust that sends a DELETE request to the
    /// OpenAI API to delete a file.
    ///
    /// Arguments:
    ///
    /// * `file_id`: The `file_id` parameter is a string that represents the unique identifier of the
    /// file you want to delete. It is used to construct the URL for the DELETE request to the OpenAI
    /// API.
    ///
    /// Returns:
    ///
    /// The function `delete_file` returns a `Result` containing either a `DeleteResponse` or a boxed
    /// dynamic error (`Box<dyn Error>`).
    #[cfg(feature = "files")]
    pub async fn delete_file(self, file_id: String) -> Result<DeleteResponse, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!("https://api.openai.com/v1/files/{}", file_id);

        let result = client
            .delete(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<DeleteResponse>(&result)
            .expect("Failed to parse delete response"))
    }

    /// The `retrieve_file` function retrieves file data from the OpenAI API using the provided file ID.
    ///
    /// Arguments:
    ///
    /// * `file_id`: The `file_id` parameter is a unique identifier for the file you want to retrieve.
    /// It is used to construct the URL for the API request to retrieve the file data.
    ///
    /// Returns:
    ///
    /// The function `retrieve_file` returns a `Result` containing either a `FileData` object or an
    /// error (`Box<dyn Error>`).
    #[cfg(feature = "files")]
    pub async fn retrieve_file(self, file_id: String) -> Result<FileData, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!("https://api.openai.com/v1/files/{}", file_id);

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FileData>(&result).expect("Failed to parse file data"))
    }

    /// The function `retrieve_file_content` retrieves the content of a file from the OpenAI API using a
    /// provided file ID.
    ///
    /// Arguments:
    ///
    /// * `file_id`: The `file_id` parameter is a unique identifier for the file you want to retrieve
    /// the content of. It is used to construct the URL for the API request to fetch the file content.
    ///
    /// Returns:
    ///
    /// The function `retrieve_file_content` returns a `Result` containing a `String` representing the
    /// content of the file with the given `file_id`. The `Ok` variant of the `Result` contains the file
    /// content as a `String`, while the `Err` variant contains a boxed dynamic error (`Box<dyn
    /// Error>`).
    #[cfg(feature = "files")]
    pub async fn retrieve_file_content(self, file_id: String) -> Result<String, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!("https://api.openai.com/v1/files/{}/content", file_id);

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(String::from_utf8(result.to_vec()).expect("Failed to parse file content"))
    }

    /// The function `create_fine_tune` sends a POST request to the OpenAI API to create a fine-tuned
    /// model and returns the retrieved data.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_fine_tune` function is of type
    /// `CreateFineTuneParameters`. It is an input parameter that contains the data required to create a
    /// fine-tune task. The specific structure and fields of the `CreateFineTuneParameters` type are not
    ///
    /// Returns:
    ///
    /// a Result object with the type FineTuneRetriveData.
    #[cfg(feature = "fine_tunes")]
    pub async fn create_fine_tune(
        self,
        parameters: CreateFineTuneParameters,
    ) -> Result<FineTuneRetriveData, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/fine-tunes");

        let result = client
            .post(url)
            .bearer_auth(self.token)
            .send_json(&parameters)
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FineTuneRetriveData>(&result)
            .expect("Failed to parse fine tune data"))
    }

    /// The function `list_fine_tunes` makes an HTTP GET request to the OpenAI API to retrieve a list of
    /// fine-tuned models.
    ///
    /// Returns:
    ///
    /// a Result object with the type FineTuneList.
    #[cfg(feature = "fine_tunes")]
    pub async fn list_fine_tunes(self) -> Result<FineTuneList, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/fine-tunes");

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(
            serde_json::from_slice::<FineTuneList>(&result)
                .expect("Failed to parse fine tune list"),
        )
    }

    /// The function retrieves fine-tune data from the OpenAI API using the provided fine-tune ID.
    ///
    /// Arguments:
    ///
    /// * `fine_tune_id`: The `fine_tune_id` parameter is a unique identifier for a specific fine-tuning
    /// job. It is used to retrieve the data associated with that fine-tuning job from the OpenAI API.
    ///
    /// Returns:
    ///
    /// a `Result` type with the success variant containing a `FineTuneRetriveData` object and the error
    /// variant containing a `Box<dyn Error>` object.
    #[cfg(feature = "fine_tunes")]
    pub async fn retrive_fine_tune(
        self,
        fine_tune_id: String,
    ) -> Result<FineTuneRetriveData, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!("https://api.openai.com/v1/fine-tunes/{}", fine_tune_id);

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FineTuneRetriveData>(&result)
            .expect("Failed to parse fine tune data"))
    }

    /// The `cancel_fine_tune` function cancels a fine-tuning process by sending a POST request to the
    /// OpenAI API.
    ///
    /// Arguments:
    ///
    /// * `fine_tune_id`: The `fine_tune_id` parameter is a unique identifier for a fine-tuning process.
    /// It is used to specify which fine-tuning process you want to cancel.
    ///
    /// Returns:
    ///
    /// a Result object with the type FineTuneRetriveData.
    #[cfg(feature = "fine_tunes")]
    pub async fn cancel_fine_tune(
        self,
        fine_tune_id: String,
    ) -> Result<FineTuneRetriveData, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!(
            "https://api.openai.com/v1/fine-tunes/{}/cancel",
            fine_tune_id
        );

        let result = client
            .post(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FineTuneRetriveData>(&result)
            .expect("Failed to parse fine tune data"))
    }

    /// The function `list_fine_tune_events` is an asynchronous function in Rust that retrieves a list
    /// of fine-tune events from the OpenAI API.
    ///
    /// Arguments:
    ///
    /// * `fine_tune_id`: The `fine_tune_id` parameter is a unique identifier for a fine-tuning job. It
    /// is used to specify which fine-tuning job's events you want to retrieve.
    ///
    /// Returns:
    ///
    /// a `Result` type with the `Ok` variant containing a `FineTuneEventList` object and the `Err`
    /// variant containing a `Box<dyn Error>` object.
    #[cfg(feature = "fine_tunes")]
    pub async fn list_fine_tune_events(
        self,
        fine_tune_id: String,
    ) -> Result<FineTuneEventList, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!(
            "https://api.openai.com/v1/fine-tunes/{}/events",
            fine_tune_id
        );

        let result = client
            .get(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FineTuneEventList>(&result)
            .expect("Failed to parse fine tune event list"))
    }

    /// The function `delete_fine_tune` sends a DELETE request to the OpenAI API to delete a fine-tuned
    /// model and returns the result as a `FineTuneDelete` object.
    ///
    /// Arguments:
    ///
    /// * `model`: The `model` parameter is a string that represents the name or ID of the fine-tuned
    /// model that you want to delete.
    ///
    /// Returns:
    ///
    /// a `Result` enum with the success variant containing a `FineTuneDelete` object or the error
    /// variant containing a `Box<dyn Error>` object.
    #[cfg(feature = "fine_tunes")]
    pub async fn delete_fine_tune(self, model: String) -> Result<FineTuneDelete, Box<dyn Error>> {
        let client = self.https_client;
        let url = format!("https://api.openai.com/v1/models/{}", model);

        let result = client
            .delete(url)
            .bearer_auth(self.token)
            .send()
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<FineTuneDelete>(&result)
            .expect("Failed to parse fine tune delete"))
    }

    /// The function `create_moderation` sends a POST request to the OpenAI API to create a text
    /// moderation task and returns the result.
    ///
    /// Arguments:
    ///
    /// * `parameters`: The `parameters` parameter in the `create_moderation` function is of type
    /// `TextModerationParameters`. It represents the input parameters for the text moderation request.
    /// The specific structure and fields of the `TextModerationParameters` type are not provided in the
    /// code snippet, so it would be
    ///
    /// Returns:
    ///
    /// a `Result` type with the success variant containing a `TextModerationResult` and the error
    /// variant containing a `Box<dyn Error>`.
    #[cfg(feature = "moderations")]
    pub async fn create_moderation(
        self,
        parameters: TextModerationParameters,
    ) -> Result<TextModerationResult, Box<dyn Error>> {
        let client = self.https_client;
        let url = String::from("https://api.openai.com/v1/moderations");

        let result = client
            .post(url)
            .bearer_auth(self.token)
            .send_json(&parameters)
            .await
            .unwrap()
            .body()
            .await
            .unwrap();

        Ok(serde_json::from_slice::<TextModerationResult>(&result)
            .expect("Failed to parse text moderation result"))
    }
}
