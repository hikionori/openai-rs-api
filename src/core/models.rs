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
        model: String,
        /// The prompt(s) to generate completions for, encoded as a string, array of strings,
        /// array of tokens, or array of token arrays.
        ///
        /// Note that <|endoftext|> is the document separator that the model sees during training,
        /// so if a prompt is not specified the model will generate as if from the beginning of a new document.
        prompt: String,
        /// The maximum number of [tokens](https://platform.openai.com/tokenizer) to generate in the completion.
        ///
        /// The token count of your prompt plus `max_tokens` cannot exceed the model's context length.
        /// [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
        /// for counting tokens.
        #[serde(skip_serializing_if = "Option::is_none")]
        max_tokens: Option<i32>,
        /// What sampling temperature to use, between 0 and 2.
        /// Higher values like 0.8 will make the output more random, while lower values
        /// like 0.2 will make it more focused and deterministic.
        ///
        /// We generally recommend altering this or `top_p` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f32>,
        /// The suffix that comes after a completion of inserted text.
        #[serde(skip_serializing_if = "Option::is_none")]
        suffix: Option<String>,
        /// An alternative to sampling with temperature, called nucleus sampling,
        /// where the model considers the results of the tokens with top_p probability mass.
        /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
        /// How many completions to generate for each prompt.
        ///
        /// Note: Because this parameter generates many completions, it can quickly consume your token quota.
        /// Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        #[serde(skip_serializing_if = "Option::is_none")]
        n: Option<i32>,
        /// Whether to stream back partial progress. If set, tokens
        /// will be sent as data-only server-sent events as they become available,
        /// with the stream terminated by a `data: [DONE]` message.
        /// [Example Python code.](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)
        #[serde(skip_serializing_if = "Option::is_none")]
        stream: Option<bool>,
        /// Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens.
        /// For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens.
        /// The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.
        ///
        /// The maximum value for logprobs is 5.
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<i32>,
        /// Up to 4 sequences where the API will stop generating further tokens.
        /// The returned text will not contain the stop sequence.
        #[serde(skip_serializing_if = "Option::is_none")]
        stop: Option<String>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
        /// they appear in the text so far, increasing the model's likelihood to talk about new topics.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        presence_penalty: Option<f32>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
        /// in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        frequency_penalty: Option<f32>,
        /// Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
        ///
        /// When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.
        ///
        /// Note: Because this parameter generates many completions, it can quickly consume your token quota.
        /// Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
        #[serde(skip_serializing_if = "Option::is_none")]
        best_of: Option<i32>,
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
        logit_bias: Option<serde_json::Value>,
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        /// Echo back the prompt in addition to the completion
        #[serde(skip_serializing_if = "Option::is_none")]
        echo: Option<bool>,
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
        /// ID of the model to use. See the
        /// [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility) table
        /// for details on which models work with the Chat API.
        model: String,
        /// A list of messages comprising the conversation so far.
        messages: Vec<Message>,
        /// A list of functions the model may generate JSON inputs for.
        #[serde(skip_serializing_if = "Option::is_none")]
        functions: Option<Vec<Function>>,
        /// Controls how the model responds to function calls. "none" means the model does not call a function,
        /// and responds to the end-user. "auto" means the model can pick between an end-user or calling a
        /// function. Specifying a particular function via `{"name":\ "my_function"}` forces the model to call
        /// that function. "none" is the default when no functions are present. "auto" is the default if functions
        /// are present.
        #[serde(skip_serializing_if = "Option::is_none")]
        function_call: Option<serde_json::Value>,
        /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output
        /// more random, while lower values like 0.2 will make it more focused and deterministic.
        ///
        /// We generally recommend altering this or `top_p` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f32>,
        /// An alternative to sampling with temperature, called nucleus sampling, where the
        /// model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        #[serde(skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
        /// How many chat completion choices to generate for each input message.
        #[serde(skip_serializing_if = "Option::is_none")]
        n: Option<i32>,
        /// If set, partial message deltas will be sent, like in ChatGPT.
        /// Tokens will be sent as data-only server-sent events as they become available,
        /// with the stream terminated by a `data: [DONE]` message. Example Python code.
        #[serde(skip_serializing_if = "Option::is_none")]
        stream: Option<bool>,
        /// Up to 4 sequences where the API will stop generating further tokens.
        #[serde(skip_serializing_if = "Option::is_none")]
        stop: Option<Vec<String>>,
        /// The maximum number of tokens to generate in the chat completion.
        #[serde(skip_serializing_if = "Option::is_none")]
        max_tokens: Option<i32>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
        /// text so far, increasing the model's likelihood to talk about new topics.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        presence_penalty: Option<f32>,
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
        /// the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        ///
        /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
        #[serde(skip_serializing_if = "Option::is_none")]
        frequency_penalty: Option<f32>,
        /// Modify the likelihood of specified tokens appearing in the completion.
        ///
        /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
        /// Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model,
        /// but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or
        /// exclusive selection of the relevant token.
        #[serde(skip_serializing_if = "Option::is_none")]
        logit_bias: Option<serde_json::Value>,
        ///A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Function {
        /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
        /// with a maximum length of 64.
        name: String,
        /// The description of what the function does.
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// The parameters the functions accepts, described as a JSON Schema object.
        /// See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples,
        /// and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
        /// documentation about the format.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "parameters")]
        params: Option<serde_json::Value>,
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
        /// The role of the messages author. One of `system`, `user`, `assistant` or `function`.
        role: String,
        /// The contents of the message. `content` is required for
        /// all messages except assistant messages with function calls.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// The name of the author of this message. `name` is required if role is `function`,
        /// and it should be the name of the function whose response is in the `content`.
        /// May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// The name and arguments of a function that should be called, as generated by the model.
        ///
        ///**Now this optional field dont support in this crate.**
        #[serde(skip_serializing_if = "Option::is_none")]
        function_call: Option<serde_json::Value>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Usage {
        prompt_tokens: i32,
        completion_tokens: i32,
        total_tokens: i32,
    }
}

pub mod images {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageCreateParameters {
        prompt: String,
        /// The number of images to generate. Must be between 1 and 10.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "n")]
        num_images: Option<i32>,
        /// The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "size")]
        image_size: Option<String>,
        /// The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_format: Option<String>, // url of b64_json
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageEditParameters {
        /// The image to edit. Must be a valid PNG file, less than 4MB, and square.
        /// If mask is not provided, image must have transparency, which will be used as the mask.
        image: String,
        /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited.
        /// Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`.
        #[serde(skip_serializing_if = "Option::is_none")]
        mask: Option<String>,
        /// A text description of the desired image(s). The maximum length is 1000 characters.
        prompt: String,
        /// The number of images to generate. Must be between 1 and 10.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "n")]
        num_images: Option<i32>,
        /// The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "size")]
        image_size: Option<String>,
        /// The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_format: Option<String>, // url of b64_json
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageVariationParameters {
        /// The image to edit. Must be a valid PNG file, less than 4MB, and square.
        /// If mask is not provided, image must have transparency, which will be used as the mask.
        image: String,
        /// The number of images to generate. Must be between 1 and 10.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "n")]
        num_images: Option<i32>,
        /// The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "size")]
        image_size: Option<String>,
        /// The format in which the generated images are returned. Must be one of `url` or `b64_json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_format: Option<String>, // url of b64_json
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageResponse {
        created: usize,
        data: Vec<ImageData>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageData {
        url: String,
    }
}

pub mod embeddings {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EmbeddingParameters {
        /// ID of the model to use. You can use the List models API to see all of your available models,
        /// or see our Model overview for descriptions of them.
        model: String,
        ///nput text to embed, encoded as a string or array of tokens. To embed multiple
        /// inputs in a single request, pass an array of strings or array of token arrays.
        /// Each input must not exceed the max input tokens for the model (8191 tokens for text-embedding-ada-002).
        input: String,
        /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        /// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        object: String,
        data: Vec<EmbeddingData>,
        model: String,
        usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct EmbeddingData {
        object: String,
        embedding: Vec<f32>,
        index: i32,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Usage {
        prompt_tokens: i32,
        total_tokens: i32,
    }
}

pub mod audio {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize)]
    pub struct TranscriptionParameters {
        /// The audio file object (not file name) to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
        file: String,
        /// ID of the model to use. Only `whisper-1` is currently available.
        model: String,
        /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
        #[serde(skip_serializing_if = "Option::is_none")]
        respone_format: Option<String>,
        /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2
        /// will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature
        /// until certain thresholds are hit.
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f32>,
        /// The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    }

    #[derive(Debug, Serialize)]
    pub struct TranslationParameters {
        /// The audio file object (not file name) to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
        file: String,
        /// ID of the model to use. Only `whisper-1` is currently available.
        model: String,
        /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
        /// The default is json.
        #[serde(skip_serializing_if = "Option::is_none")]
        respone_format: Option<String>,
        /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2
        /// will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature
        /// until certain thresholds are hit.
        /// The default is 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f32>,
    }

    #[derive(Debug, Deserialize)]
    pub struct TextResponse {
        text: String,
    }
}

pub mod files {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FileList {
        data: Vec<FileData>,
        object: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FileData {
        id: String,
        object: String,
        bytes: u32,
        created_at: u64,
        filename: String,
        purpose: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FileUpload {
        /// Name of the [JSON Lines](https://jsonlines.readthedocs.io/en/latest/) file to be uploaded.
        ///
        /// If the purpose is set to "fine-tune", each line is a JSON record with "prompt" and "completion"
        /// fields representing your [training examples.](https://platform.openai.com/docs/guides/fine-tuning/prepare-training-data)
        file: String,
        /// The intended purpose of the uploaded documents.
        ///
        /// Use "fine-tune" for Fine-tuning. This allows us to validate the format of the uploaded file.
        purpose: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct DeleteResponse {
        id: String,
        object: String,
        deleted: bool,
    }
}

pub mod fine_tunes {}

pub mod moderations {}
