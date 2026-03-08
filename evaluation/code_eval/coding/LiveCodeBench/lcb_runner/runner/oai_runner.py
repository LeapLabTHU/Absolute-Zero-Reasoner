import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    @staticmethod
    def _build_client():
        forge_api_key = os.getenv("FORGE_API_KEY")
        if forge_api_key:
            return OpenAI(
                api_key=forge_api_key,
                base_url=os.getenv("FORGE_API_BASE")
                or "https://api.forge.tensorblock.co/v1",
            )
        return OpenAI(api_key=os.getenv("OPENAI_KEY"))

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client = self._build_client()
        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        try:
            response = self.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]
