"""Shared EgoNormia prompt templates used by preprocessing and evaluation."""

VIDEO_TAG_PREFIX = "<video>\n"

ACTION_PROMPT_TEMPLATE = """\
The following images from a first-person perspective video depict a person in a given situation.

Given the below list of behaviors, choose the single most normatively relevant or appropriate action to perform next.

Think carefully about your response and reason over it in the output.

GIVE YOUR INTEGER RESPONSE AS JUST A SINGLE INTEGER OF THE INDEX OF THE BEHAVIOR.

Behavior options:
{behavior_options}"""

V6_ACTION_PROMPT_TEMPLATE = """\
The following images from a first-person perspective video depict a person in a given situation.

Given the below list of behaviors, choose the single most normatively relevant or appropriate action to perform next.

Respond with the behavior index followed by the behavior text.

The only possible indices are 1, 2, 3, 4, or 5.

Behavior options:
{behavior_options}"""

JUSTIFICATION_PROMPT_TEMPLATE = """\
The following images from a first-person perspective video depict a person performing some action.

{behavior} is selected as the most normatively relevant or appropriate action for the person to perform in the given situation.

Your task is to now choose the most normatively correct justification that best supports this behavior, based on context and commonsense norms.

Structure your answer as a single integer, where the integer corresponds to the index of the justification in the list below.

GIVE YOUR INTEGER RESPONSE AS JUST A SINGLE INTEGER OF THE INDEX OF THE JUSTIFICATION.

The only possible integers you should output are 1, 2, 3, 4, or 5.

Justification options:
{justification_options}"""

V6_JUSTIFICATION_PROMPT_TEMPLATE = """\
The following images from a first-person perspective video depict a person performing some action.

{behavior} is selected as the most normatively relevant or appropriate action for the person to perform in the given situation.

Your task is to choose the most normatively correct justification that best supports this behavior.

Respond with the justification index followed by the justification text.

The only possible indices are 1, 2, 3, 4, or 5.

Justification options:
{justification_options}"""

SENSIBILITY_PROMPT_TEMPLATE = """\
The following images from a first-person perspective video depict a person in a given situation.

Given the below behaviors, choose ALL sensible actions to perform in the given situation, based on context and commonsense norms.

Structure your answer as one Python list of integers, where each integer corresponds to the indices of the behaviors in the list below, from 1 to 5.
An empty list is acceptable if no behavior is sensible.

DO NOT WRITE ANY OTHER TEXT IN YOUR RESPONSE, JUST A PYTHON LIST OF INTEGERS.

Behavior options:
{behavior_options}"""

V6_SENSIBILITY_PROMPT_TEMPLATE = """\
The following images from a first-person perspective video depict a person in a given situation.

Given the below behaviors, choose ALL sensible actions to perform in the given situation.

First output one Python list of sensible indices (from 1 to 5), then list each sensible behavior on a new line as "{index}. {text}".

Behavior options:
{behavior_options}"""

REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""


def with_video_tag(template: str) -> str:
    """Attach training-time <video> tag prefix."""
    return f"{VIDEO_TAG_PREFIX}{template}"
