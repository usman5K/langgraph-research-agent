"""A scripted chat model so the whole agent loop runs offline, with no API key."""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Returns the given AIMessages in order; records what it was shown."""

    responses: list[AIMessage]
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._calls.append(list(messages))
        template = self.responses[min(len(self._calls), len(self.responses)) - 1]
        # fresh copy per turn (unique ids) so repeated responses append, not overwrite
        n = len(self._calls)
        calls = [{**c, "id": f"{c['id']}_{n}"} for c in template.tool_calls]
        message = AIMessage(content=template.content, tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def scripted():
    return ScriptedChatModel
