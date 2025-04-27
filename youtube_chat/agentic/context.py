from agentic.starter import UserInfo
from pydantic import BaseModel
from pydantic_models import SegmentedTranscript


class Context(BaseModel):
    user_info: UserInfo | None = None
    segmented_transcript: SegmentedTranscript | None = None
    current_selected_segment: str | None = None
