import openai
import pytest
import pytest_asyncio

import quartapp

from . import mock_cred


@pytest.fixture
def mock_openai_chatcompletion(monkeypatch):
    class AsyncChatCompletionIterator:
        def __init__(self, reasoning: str, answer: str):
            self.chunk_index = 0
            self.chunks = [
                openai.types.chat.ChatCompletionChunk(
                    object="chat.completion.chunk",
                    choices=[],
                    id="",
                    created=0,
                    model="",
                    prompt_filter_results=[
                        {
                            "prompt_index": 0,
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                            },
                        }
                    ],
                )
            ]
            reasoning_deltas = reasoning.split(" ")
            for reasoning_index, reasoning_delta in enumerate(reasoning_deltas):
                # Text completion chunks include whitespace, so we need to add it back in
                if reasoning_index > 0:
                    answer_delta = " " + reasoning_delta
                self.chunks.append(
                    openai.types.chat.ChatCompletionChunk(
                        id="test-123",
                        object="chat.completion.chunk",
                        choices=[
                            openai.types.chat.chat_completion_chunk.Choice(
                                delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(
                                    role=None, reasoning_content=reasoning_delta
                                ),
                                finish_reason=None,
                                index=0,
                                logprobs=None,
                                # Only Azure includes content_filter_results
                                content_filter_results={
                                    "hate": {"filtered": False, "severity": "safe"},
                                    "self_harm": {"filtered": False, "severity": "safe"},
                                    "sexual": {"filtered": False, "severity": "safe"},
                                    "violence": {"filtered": False, "severity": "safe"},
                                },
                            )
                        ],
                        created=1703462735,
                        model="DeepSeek-R1",
                    )
                )

            answer_deltas = answer.split(" ")
            for answer_index, answer_delta in enumerate(answer_deltas):
                # Text completion chunks include whitespace, so we need to add it back in
                if answer_index > 0:
                    answer_delta = " " + answer_delta
                self.chunks.append(
                    openai.types.chat.ChatCompletionChunk(
                        id="test-123",
                        object="chat.completion.chunk",
                        choices=[
                            openai.types.chat.chat_completion_chunk.Choice(
                                delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(
                                    role=None, content=answer_delta
                                ),
                                finish_reason=None,
                                index=0,
                                logprobs=None,
                                # Only Azure includes content_filter_results
                                content_filter_results={
                                    "hate": {"filtered": False, "severity": "safe"},
                                    "self_harm": {"filtered": False, "severity": "safe"},
                                    "sexual": {"filtered": False, "severity": "safe"},
                                    "violence": {"filtered": False, "severity": "safe"},
                                },
                            )
                        ],
                        created=1703462735,
                        model="DeepSeek-R1",
                    )
                )
            self.chunks.append(
                openai.types.chat.ChatCompletionChunk(
                    id="test-123",
                    object="chat.completion.chunk",
                    choices=[
                        openai.types.chat.chat_completion_chunk.Choice(
                            delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(content=None, role=None),
                            index=0,
                            finish_reason="stop",
                            # Only Azure includes content_filter_results
                            content_filter_results={},
                        )
                    ],
                    created=1703462735,
                    model="DeepSeek-R1",
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.chunk_index < len(self.chunks):
                next_chunk = self.chunks[self.chunk_index]
                self.chunk_index += 1
                return next_chunk
            else:
                raise StopAsyncIteration

    async def mock_acreate(*args, **kwargs):
        # Only mock a stream=True completion
        last_message = kwargs.get("messages")[-1]["content"]
        if last_message == "What is the capital of France?":
            return AsyncChatCompletionIterator("hmm", "The capital of France is Paris.")
        elif last_message == "What is the capital of Germany?":
            return AsyncChatCompletionIterator("hmm", "The capital of Germany is Berlin.")
        else:
            raise ValueError(f"Unexpected message: {last_message}")

    monkeypatch.setattr("openai.resources.chat.AsyncCompletions.create", mock_acreate)


@pytest.fixture
def mock_defaultazurecredential(monkeypatch):
    monkeypatch.setattr("azure.identity.aio.AzureDeveloperCliCredential", mock_cred.MockAzureCredential)
    monkeypatch.setattr("azure.identity.aio.ManagedIdentityCredential", mock_cred.MockAzureCredential)


@pytest_asyncio.fixture
async def client(monkeypatch, mock_openai_chatcompletion, mock_defaultazurecredential):
    monkeypatch.setenv("AZURE_INFERENCE_ENDPOINT", "test-deepseek-service.ai.azure.com")
    monkeypatch.setenv("AZURE_TENANT_ID", "test-tenant-id")

    quart_app = quartapp.create_app(testing=True)

    async with quart_app.test_app() as test_app:
        quart_app.config.update({"TESTING": True})

        yield test_app.test_client()
