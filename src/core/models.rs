#![allow(dead_code)]

pub mod list_models {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct ModelPermission {
        /// The ID of the permission.
        pub id: String,
        /// The type of object returned by the API. In this case, it will always be "model_permission".
        pub object: String,
        /// The Unix timestamp (in seconds) when the permission was created.
        pub created: i64,
        /// Whether the permission allows creating engines.
        pub allow_create_engine: bool,
        /// Whether the permission allows sampling.
        pub allow_sampling: bool,
        /// Whether the permission allows log probabilities.
        pub allow_logprobs: bool,
        /// Whether the permission allows search indices.
        pub allow_search_indices: bool,
        /// Whether the permission allows viewing.
        pub allow_view: bool,
        /// Whether the permission allows fine-tuning.
        pub allow_fine_tuning: bool,
        /// The ID of the organization that the permission belongs to.
        pub organization: String,
        /// The ID of the group that the permission belongs to.
        pub group: Option<String>,
        /// Whether the permission is blocking.
        pub is_blocking: bool,
    }

    #[derive(Debug, Deserialize)]
    pub struct Model {
        /// The ID of the model.
        pub id: String,
        /// The type of object returned by the API. In this case, it will always be "model".
        pub object: String,
        /// The Unix timestamp (in seconds) when the model was created.
        pub created: i64,
        /// The ID of the organization that owns the model.
        pub owned_by: String,
        /// A list of `ModelPermission` objects representing the permissions for the model.
        pub permission: Vec<ModelPermission>,
        /// The ID of the root model that this model was created from.
        pub root: String,
        /// The ID of the parent model that this model was created from.
        pub parent: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    pub struct ModelList {
        /// The type of object returned by the API. In this case, it will always be "list".
        pub object: String,
        /// A vector of `Model` objects representing the models returned by the API.
        pub data: Vec<Model>,
    }
}

pub mod edits {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize)]
    pub struct EditParameters {
        /// ID of the model to use. You can use the `text-davinci-edit-001` or `code-davinci-edit-001` model with this endpoint.
        model: String,
        /// The input text to use as a starting point for the edit.
        input: String,
        /// The instruction that tells the model how to edit the prompt.
        instructions: String,
        /// How many edits to generate for the input and instruction.
        #[serde(skip_serializing_if = "Option::is_none")]
        n_of_edits: Option<i32>,
        /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        ///
        /// We generally recommend altering this or `top_p` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f32>,
        /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
    }

    #[derive(Debug, Deserialize)]
    pub struct EditResponse {
        /// The type of object returned by the API. In this case, it will always be "text_completion".
        object: String,
        /// The Unix timestamp (in seconds) when the completion was generated.
        created: i64,
        /// A list of `Choice` objects representing the generated completions.
        choices: Vec<Choice>,
        /// An object containing information about the number of tokens used in the prompt and generated completion.
        usage: Usage,
    }

    #[derive(Debug, Deserialize)]
    pub struct Choice {
        /// The generated text for this choice.
        text: String,
        /// The index of this choice in the list of choices returned by the API.
        index: i32,
    }

    #[derive(Debug, Deserialize)]
    pub struct Usage {
        /// The number of tokens in the prompt.
        prompt_tokens: i32,
        /// The number of tokens in the generated completion.
        completion_tokens: i32,
        /// The total number of tokens used (prompt + completion).
        total_tokens: i32,
    }
}

pub mod completions {

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize)]
    pub struct CompletionParameters {
        /// ID of the model to use. You can use the List models API to see all of your available models,
        /// or see our [Model overview](https://platform.openai.com/docs/models/overview) for descriptions of them.
        ///
        /// List models example:
        /// ```rust
        /// use openai_rs_api::core::{OpenAI, models::list_models::ModelList};
        /// use tokio;
        ///
        /// #[tokio::main]
        /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
        ///     let openai = OpenAI::new("your_api_key", "your_organization_id");
        ///     let models: ModelList = openai.list_models().await?;
        ///     println!("{:#?}", models);
        /// }
        /// ```
        ///
        pub model: String,
        /// The prompt(s) to generate completions for, encoded as a string, array of strings,
        /// array of tokens, or array of token arrays.
        ///
        /// Note that <|endoftext|> is the document separator that the model sees during training,
        /// so if a prompt is not specified the model will generate as if from the beginning of a new document.
        pub prompt: String,
        /// The maximum number of [tokens](https://platform.openai.com/tokenizer) to generate in the completion.
        ///
        /// The token count of your prompt plus `max_tokens` cannot exceed the model's context length.
        /// [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
        /// for counting tokens.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<i32>,
        /// What sampling temperature to use, between 0 and 2.
        /// Higher values like 0.8 will make the output more random, while lower values
        /// like 0.2 will make it more focused and deterministic.
        ///
        /// We generally recommend altering this or `top_p` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        /// The suffix that comes after a completion of inserted text.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub suffix: Option<String>,
        /// An alternative to sampling with temperature, called nucleus sampling,
        /// where the model considers the results of the tokens with top_p probability mass.
        /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        /// How many completions to generate for each prompt.
        ///
        /// Note: Because this parameter generates many completions, it can quickly consume your token quota.
        /// Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<i32>,
        /// Whether to stream back partial progress. If set, tokens
        /// will be sent as data-only server-sent events as they become available,
        /// with the stream terminated by a `data: [DONE]` message.
        /// [Example Python code.](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        /// Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens.
        /// For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens.
        /// The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.
        ///
        /// The maximum value for logprobs is 5.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<i32>,
        /// Up to 4 sequences where the API will stop generating further tokens.
        /// The returned text will not contain the stop sequence.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<String>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
        /// they appear in the text so far, increasing the model's likelihood to talk about new topics.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f32>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
        /// in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f32>,
        /// Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
        ///
        /// When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.
        ///
        /// Note: Because this parameter generates many completions, it can quickly consume your token quota.
        /// Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub best_of: Option<i32>,
        /// Modify the likelihood of specified tokens appearing in the completion.
        ///
        /// Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer)
        /// to an associated bias value from -100 to 100. You can use this [tokenizer tool](https://platform.openai.com/tokenizer?view=bpe)
        /// (which works for both GPT-2 and GPT-3) to convert text to token IDs.
        /// Mathematically, the bias is added to the logits generated by the model prior to sampling.
        /// The exact effect will vary per model, but values between -1 and 1 should decrease or
        /// increase likelihood of selection; values like -100 or 100 should result in a ban or
        /// exclusive selection of the relevant token.
        ///
        /// As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token from being generated.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logit_bias: Option<serde_json::Value>,
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
        /// Echo back the prompt in addition to the completion
        #[serde(skip_serializing_if = "Option::is_none")]
        pub echo: Option<bool>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CompletionResponse {
        /// The unique identifier for the completion request.
        pub id: String,
        /// The type of object, which is always "text_completion".
        pub object: String,
        /// The Unix timestamp (in seconds) when the completion request was created.
        pub created: i64,
        /// The ID of the model used to generate the completion.
        pub model: String,
        /// A vector of `CompletionChoice` objects, each representing a possible completion.
        pub choices: Vec<CompletionChoice>,
        /// An object containing usage statistics for the completion request.
        pub usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CompletionChoice {
        /// The generated text for this completion choice.
        pub text: String,
        /// The index of this completion choice in the list of all possible choices.
        pub index: i32,
        /// The log probabilities of the tokens in the generated text.
        /// If the `logprobs` parameter was not set in the request, this field will be `None`.
        pub logprobs: Option<i32>,
        /// The reason why the completion was finished.
        /// Possible values are "stop", "length", "temperature", "top_p", "nucleus_sampling", and "incomplete".
        pub finish_reason: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Usage {
        /// prompt_tokens: an integer representing the number of tokens in the prompt used for the completion request.
        pub prompt_tokens: i32,
        /// completion_tokens: an integer representing the number of tokens in the generated completion text.
        pub completion_tokens: i32,
        /// total_tokens: an integer representing the total number of tokens used in the completion request, including both the prompt and the generated completion text.
        pub total_tokens: i32,
    }
}

pub mod chat {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatParameters {
        /// ID of the model to use. See the
        /// [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility) table
        /// for details on which models work with the Chat API.
        pub model: String,
        /// A list of messages comprising the conversation so far.
        pub messages: Vec<Message>,
        /// A list of functions the model may generate JSON inputs for.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub functions: Option<Vec<Function>>,
        /// Controls how the model responds to function calls. "none" means the model does not call a function,
        /// and responds to the end-user. "auto" means the model can pick between an end-user or calling a
        /// function. Specifying a particular function via `{"name":\ "my_function"}` forces the model to call
        /// that function. "none" is the default when no functions are present. "auto" is the default if functions
        /// are present.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub function_call: Option<serde_json::Value>,
        /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output
        /// more random, while lower values like 0.2 will make it more focused and deterministic.
        ///
        /// We generally recommend altering this or `top_p` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        /// An alternative to sampling with temperature, called nucleus sampling, where the
        /// model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        /// How many chat completion choices to generate for each input message.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<i32>,
        /// If set, partial message deltas will be sent, like in ChatGPT.
        /// Tokens will be sent as data-only server-sent events as they become available,
        /// with the stream terminated by a `data: [DONE]` message. Example Python code.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        /// Up to 4 sequences where the API will stop generating further tokens.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<Vec<String>>,
        /// The maximum number of tokens to generate in the chat completion.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<i32>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
        /// text so far, increasing the model's likelihood to talk about new topics.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f32>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
        /// the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f32>,
        /// Modify the likelihood of specified tokens appearing in the completion.
        ///
        /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
        /// Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model,
        /// but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or
        /// exclusive selection of the relevant token.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logit_bias: Option<serde_json::Value>,
        ///A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Function {
        /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
        /// with a maximum length of 64.
        pub name: String,
        /// The description of what the function does.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        /// The parameters the functions accepts, described as a JSON Schema object.
        /// See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples,
        /// and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
        /// documentation about the format.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "parameters")]
        pub params: Option<serde_json::Value>,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct ChatResponse {
        /// The unique identifier for this chat response.
        pub id: String,
        /// The type of object, which is always "text_completion".
        pub object: String,
        /// The Unix timestamp (in seconds) when this chat response was created.
        pub created: i64,
        /// A vector of `CompletionChoice` structs, representing the different choices for the chat response.
        pub choices: Vec<CompletionChoice>,
        /// An object containing usage information for this API request.
        pub usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct CompletionChoice {
        /// The index of this choice in the list of choices returned by the API.
        pub index: i32,
        /// The message generated by the API for this choice.
        pub message: Message,
        /// The reason why the API stopped generating further tokens for this choice.
        pub finish_reason: String,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct Message {
        /// The role of the messages author. One of `system`, `user`, `assistant` or `function`.
        pub role: String,
        /// The contents of the message. `content` is required for
        /// all messages except assistant messages with function calls.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        /// The name of the author of this message. `name` is required if role is `function`,
        /// and it should be the name of the function whose response is in the `content`.
        /// May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        /// The name and arguments of a function that should be called, as generated by the model.
        ///
        ///**Now this optional field dont support in this crate.**
        #[serde(skip_serializing_if = "Option::is_none")]
        pub function_call: Option<serde_json::Value>,
    }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct Usage {
        pub prompt_tokens: i32,
        pub completion_tokens: i32,
        pub total_tokens: i32,
    }
}

pub mod images {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageCreateParameters {
        pub prompt: String,
        /// The number of images to generate. Must be between 1 and 10.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "n")]
        pub num_images: Option<i32>,
        /// The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "size")]
        pub image_size: Option<String>,
        /// The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<String>, // url of b64_json
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageEditParameters {
        /// The image to edit. Must be a valid PNG file, less than 4MB, and square.
        /// If mask is not provided, image must have transparency, which will be used as the mask.
        pub image: String,
        /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited.
        /// Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mask: Option<String>,
        /// A text description of the desired image(s). The maximum length is 1000 characters.
        pub prompt: String,
        /// The number of images to generate. Must be between 1 and 10.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "n")]
        pub num_images: Option<i32>,
        /// The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "size")]
        pub image_size: Option<String>,
        /// The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<String>, // url of b64_json
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageVariationParameters {
        /// The image to edit. Must be a valid PNG file, less than 4MB, and square.
        /// If mask is not provided, image must have transparency, which will be used as the mask.
        pub image: String,
        /// The number of images to generate. Must be between 1 and 10.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "n")]
        pub num_images: Option<i32>,
        /// The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "size")]
        pub image_size: Option<String>,
        /// The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<String>, // url of b64_json
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageResponse {
        /// The timestamp (in seconds since the Unix epoch) when the request was made.
        pub created: usize,
        /// A vector of ImageData structs containing the URLs of the generated images.
        pub data: Vec<ImageData>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageData {
        /// The URL of the generated image.
        pub url: String,
    }
}

pub mod embeddings {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EmbeddingParameters {
        /// ID of the model to use. You can use the List models API to see all of your available models,
        /// or see our Model overview for descriptions of them.
        pub model: String,
        ///nput text to embed, encoded as a string or array of tokens. To embed multiple
        /// inputs in a single request, pass an array of strings or array of token arrays.
        /// Each input must not exceed the max input tokens for the model (8191 tokens for text-embedding-ada-002).
        pub input: String,
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        /// A string representing the type of object returned. In this case, it should always be "embedding".
        pub object: String,
        /// A vector of `EmbeddingData` representing the embedding of the input text.
        pub data: Vec<EmbeddingData>,
        /// ID of the model used for the embedding.
        pub model: String,
        /// An object containing information about the API usage for the request.
        pub usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EmbeddingData {
        /// object: A string representing the type of object returned. In this case, it should always be "embedding".
        pub object: String,
        /// embedding: A vector of 32-bit floating point numbers representing the embedding of the input text. The length of the vector depends on the model used for the embedding.
        pub embedding: Vec<f32>,
        /// index: An integer representing the index of the input text in the request. This is useful when multiple inputs are passed in a single request.
        pub index: i32,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Usage {
        /// prompt_tokens: An integer representing the number of tokens used in the prompt for the API request.
        pub prompt_tokens: i32,
        /// total_tokens: An integer representing the total number of tokens used in the API request, including the prompt tokens.
        pub total_tokens: i32,
    }
}
pub mod audio {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize)]
    pub struct TranscriptionParameters {
        /// The audio file object (not file name) to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
        pub file: String,
        /// ID of the model to use. Only `whisper-1` is currently available.
        pub model: String,
        /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prompt: Option<String>,
        /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub respone_format: Option<String>,
        /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2
        /// will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature
        /// until certain thresholds are hit.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        /// The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub language: Option<String>,
    }

    #[derive(Debug, Serialize)]
    pub struct TranslationParameters {
        /// The audio file object (not file name) to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
        pub file: String,
        /// ID of the model to use. Only `whisper-1` is currently available.
        pub model: String,
        /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prompt: Option<String>,
        /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
        /// The default is json.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub respone_format: Option<String>,
        /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2
        /// will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature
        /// until certain thresholds are hit.
        /// The default is 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
    }

    #[derive(Debug, Deserialize)]
    pub struct TextResponse {
        /// The generated text from the OpenAI API.
        pub text: String,
    }
}

pub mod files {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FileList {
        /// A vector of `FileData` objects representing the files returned by the API.
        pub data: Vec<FileData>,
        /// A string representing the object type returned by the API. This should always be "list".
        pub object: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FileData {
        /// The unique identifier for the file.
        pub id: String,
        /// The type of object, which should always be "file".
        pub object: String,
        /// The size of the file in bytes.
        pub bytes: u32,
        /// The Unix timestamp (in seconds) when the file was created.
        pub created_at: u64,
        /// The name of the file.
        pub filename: String,
        /// The intended purpose of the file.
        pub purpose: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FileUpload {
        /// Name of the [JSON Lines](https://jsonlines.readthedocs.io/en/latest/) file to be uploaded.
        ///
        /// If the purpose is set to "fine-tune", each line is a JSON record with "prompt" and "completion"
        /// fields representing your [training examples.](https://platform.openai.com/docs/guides/fine-tuning/prepare-training-data)
        pub file: String,
        /// The intended purpose of the uploaded documents.
        ///
        /// Use "fine-tune" for Fine-tuning. This allows us to validate the format of the uploaded file.
        pub purpose: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct DeleteResponse {
        /// The unique identifier for the deleted object.
        pub id: String,
        /// The type of object that was deleted.
        pub object: String,
        /// A boolean indicating whether the object was successfully deleted.
        pub deleted: bool,
    }
}

pub mod fine_tunes {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct CreateFineTuneParameters {
        /// The ID of an uploaded file that contains training data.
        pub training_file: String,
        /// The ID of an uploaded file that contains validation data.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub validation_file: Option<String>,
        /// The name of the base model to use for fine-tuning.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        /// The number of epochs to train the model for.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub epochs: Option<u32>,
        /// The batch size to use for training.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub batch_size: Option<u32>,
        /// The learning rate multiplier to use for training.
        /// The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub learning_rate_multiplier: Option<f32>,
        /// The weight to use for loss on the prompt tokens. This controls how much the model tries
        /// to learn to generate the prompt (as compared to the completion which always has a weight of 1.0),
        /// and can add a stabilizing effect to training when completions are short.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prompt_loss_weight: Option<f32>,
        /// If set, we calculate classification-specific metrics such as accuracy and F-1 score using
        /// the validation set at the end of every epoch. These metrics can be viewed in the results file.
        ///
        /// In order to compute classification metrics, you must provide a `validation_file`.
        /// Additionally, you must specify `classification_n_classes` for multiclass classification or
        /// `classification_positive_class` for binary classification.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub compute_classification_metrics: Option<bool>,
        /// The number of classes in a classification task.
        ///
        /// This parameter is required for multiclass classification.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub classification_n_classes: Option<u32>,
        /// The positive class in binary classification.
        ///
        /// This parameter is needed to generate precision, recall,
        /// and F1 metrics when doing binary classification.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub classification_positive_class: Option<String>,
        /// If this is provided, we calculate F-beta scores at the specified beta values.
        /// The F-beta score is a generalization of F-1 score. This is only used for binary classification.
        ///
        /// With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight.
        /// A larger beta score puts more weight on recall and less on precision. A smaller beta score puts
        /// more weight on precision and less on recall.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub classification_beta: Option<f32>,
        ///A string of up to 40 characters that will be added to your fine-tuned model name.
        ///
        /// For example, a suffix of "custom-model-name" would produce a model name like ada:ft-your-org:custom-model-name-2022-02-15-04-21-04.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub suffix: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FineTuneList {
        /// The object type, which is always "list".
        pub object: String,
        /// A vector of `FineTuneData` structs representing the fine-tuned models.
        pub data: Vec<FineTuneListData>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FineTuneRetriveData {
        /// The ID of the fine-tuned model.
        pub id: String,
        /// The object type, which is always "fine_tune".
        pub object: String,
        /// The name of the base model that was fine-tuned.
        pub model: String,
        /// The Unix timestamp (in seconds) when the fine-tuned model was created.
        pub created_at: i64,
        /// The vector of `FineTuneEvent` structs representing the events that occurred during fine-tuning.
        pub events: Vec<FineTuneEvent>,
        /// The ID of the fine-tuned model that was created.
        pub fine_tuned_model: Option<String>,
        /// The hyperparameters used for fine-tuning the model.
        pub hyperparams: FineTuneHyperparams,
        /// The ID of the organization that created the fine-tuned model.
        pub organization_id: String,
        /// A vector of URLs pointing to the result files generated during fine-tuning.
        pub result_files: Vec<String>,
        /// The status of the fine-tuned model.
        pub status: String,
        /// A vector of `FineTuneFiles` structs representing the validation files used during fine-tuning.
        pub validation_files: Vec<FineTuneFiles>,
        /// A vector of `FineTuneFiles` structs representing the training files used during fine-tuning.
        pub training_files: Vec<FineTuneFiles>,
        /// The Unix timestamp (in seconds) when the fine-tuned model was last updated.
        pub updated_at: i64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FineTuneListData {
        /// The ID of the fine-tuned model.
        pub id: String,
        /// The object type, which is always "fine_tune".
        pub object: String,
        /// The name of the base model that was fine-tuned.
        pub model: String,
        /// The Unix timestamp (in seconds) when the fine-tuned model was created.
        pub created_at: i64,
        /// The ID of the fine-tuned model that was created.
        pub fine_tuned_model: Option<String>,
        /// The hyperparameters used for fine-tuning the model.
        pub hyperparams: FineTuneHyperparams,
        /// The ID of the organization that created the fine-tuned model.
        pub organization_id: String,
        /// A vector of URLs pointing to the result files generated during fine-tuning.
        pub result_files: Vec<String>,
        /// The status of the fine-tuned model.
        pub status: String,
        /// A vector of `FineTuneFiles` structs representing the validation files used during fine-tuning.
        pub validation_files: Vec<FineTuneFiles>,
        /// A vector of `FineTuneFiles` structs representing the training files used during fine-tuning.
        pub training_files: Vec<FineTuneFiles>,
        /// The Unix timestamp (in seconds) when the fine-tuned model was last updated.
        pub updated_at: i64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    /// A struct representing the hyperparameters used for fine-tuning a model.
    pub struct FineTuneHyperparams {
        /// The batch size used during fine-tuning.
        pub batch_size: u32,
        /// The number of epochs used during fine-tuning.
        pub epochs: u32,
        /// A multiplier applied to the learning rate during fine-tuning.
        pub learning_rate_multiplier: f32,
        /// The weight given to the prompt loss during fine-tuning.
        pub prompt_loss_weight: f32,
    }

    #[derive(Debug, Serialize, Deserialize)]
    /// A struct representing a file used during fine-tuning a model.
    pub struct FineTuneFiles {
        /// The ID of the file.
        pub id: String,
        /// The object type, which is always "file".
        pub object: String,
        /// The size of the file in bytes.
        pub bytes: u32,
        /// The Unix timestamp (in seconds) when the file was created.
        pub created_at: i64,
        /// The name of the file.
        pub filename: String,
        /// The purpose of the file, which can be "training" or "validation".
        pub purpose: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FineTuneEventList {
        /// The object type, which is always "list".
        pub object: String,
        /// A vector of `FineTuneEvent` structs representing the fine-tuned events.
        pub data: Vec<FineTuneEvent>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    /// A struct representing a fine-tuned event.
    pub struct FineTuneEvent {
        /// The object type, which is always "fine_tune_event".
        pub object: String,
        /// The Unix timestamp (in seconds) when the fine-tuned event was created.
        pub created_at: i64,
        /// The level of the fine-tuned event, which can be "info", "warning", or "error".
        pub level: String,
        /// The message associated with the fine-tuned event.
        pub message: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FineTuneDelete {
        /// The ID of the fine-tuned model that was deleted.
        pub id: String,
        /// The object type, which is always "fine_tune".
        pub object: String,
        /// A boolean indicating whether the fine-tuned model was successfully deleted.
        pub deleted: bool,
    }
}

pub mod moderations {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TextModerationParameters {
        /// The ID of the model to use for moderation.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        /// The text to moderate.
        pub input: String
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TextModerationResult {
        /// The ID of the moderation result.
        pub id: String,
        /// The name of the model used for moderation.
        pub model: String,
        /// The moderation results.
        pub results: Vec<TextModerationCategory>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TextModerationCategory {
        /// Whether the text was flagged for this category.
        pub flagged: bool,
        /// The categories and their corresponding boolean values.
        pub categories: TextModerationCategoryValues,
        /// The scores for each category.
        pub category_scores: TextModerationCategoryScores,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TextModerationCategoryValues {
        /// Whether the text was flagged for sexual content.
        pub sexual: bool,
        /// Whether the text was flagged for hate speech.
        pub hate: bool,
        /// Whether the text was flagged for harassment.
        pub harassment: bool,
        /// Whether the text was flagged for self-harm.
        pub self_harm: bool,
        /// Whether the text was flagged for sexual content involving minors.
        pub sexual_minors: bool,
        /// Whether the text was flagged for hate speech with threatening language.
        pub hate_threatening: bool,
        /// Whether the text was flagged for graphic violence.
        pub violence_graphic: bool,
        /// Whether the text was flagged for self-harm with intent.
        pub self_harm_intent: bool,
        /// Whether the text was flagged for self-harm instructions.
        pub self_harm_instructions: bool,
        /// Whether the text was flagged for harassment with threatening language.
        pub harassment_threatening: bool,
        /// Whether the text was flagged for violence.
        pub violence: bool,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct TextModerationCategoryScores {
        /// The score for sexual content.
        pub sexual: f64,
        /// The score for hate speech.
        pub hate: f64,
        /// The score for harassment.
        pub harassment: f64,
        /// The score for self-harm.
        pub self_harm: f64,
        /// The score for sexual content involving minors.
        pub sexual_minors: f64,
        /// The score for hate speech with threatening language.
        pub hate_threatening: f64,
        /// The score for graphic violence.
        pub violence_graphic: f64,
        /// The score for self-harm with intent.
        pub self_harm_intent: f64,
        /// The score for self-harm instructions.
        pub self_harm_instructions: f64,
        /// The score for harassment with threatening language.
        pub harassment_threatening: f64,
        /// The score for violence.
        pub violence: f64,
    }
}
