import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class AOAIConfig:
    api_key = os.getenv("AOAI_API_KEY")
    api_version = os.getenv("AOAI_API_VERSION")
    api_endpoint = os.getenv("AOAI_API_ENDPOINT")

    def __init__(self):
        if not all([self.api_key, self.api_version, self.api_endpoint]):
            raise ValueError("Missing required Azure OpenAI environment variables.")


class label(AOAIConfig):
    def __init__(self, context_prompt_path: Path, llm_deployed_model: str = "cv-anomaly-gpt4o-mini"):
        super().__init__()
        self.context_prompt_path = context_prompt_path
        self.llm_model = llm_deployed_model

        self.context_prompt = self._get_context_prompt()  # str로 저장
        self.client = self._client()

    def _get_context_prompt(self) -> str:
        if not isinstance(self.context_prompt_path, Path):
            raise ValueError("context_prompt_path must be a pathlib.Path")
        if not self.context_prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.context_prompt_path}")
        return self.context_prompt_path.read_text(encoding="utf-8")

    def _client(self):
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint,
        )

    def run(self, base64_image: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.context_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty LLM response")
        return content

if __name__ == "__main__":
    print("endpoint =", AOAIConfig.api_endpoint)
    print("model/deployment =", "cv-anomaly-gpt4o-mini")
