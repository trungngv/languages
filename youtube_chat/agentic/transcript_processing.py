"""
Module/tool for processing a Youtube transcript.

Concepts demonstrated:
- Calling agents from a tool (showing the dynamic collaboration between agents and tools)
- Using the Agent interface as a nice abstraction over a single LLM call
- Using contextWrapper to pass additional information to the agent, making it dynamic prompt
- Using output_type to pass the expected output type to the agent
- Using database to store/cache data for faster processing
"""

import logging

from agentic.starter import UserInfo
from agents import Agent, RunContextWrapper, Runner
from services.database import VideoDatabase
from services.youtube import YouTubeTranscriptDownloader

from youtube_chat.pydantic_models import (
    SegmentedTranscript,
    Transcript,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pattern: define the agent's dynamic prompt using input values from context inside the contextWrapper
def review_transcript_instructions(
    ctxWrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    return f"""
Review the following transcript and:
1. Determine the language
2. Add proper punctuation if missing
3. Format the text for better readability

Transcript:
{ctxWrapper.context}
"""


review_transcript_agent = Agent(
    name="Review Transcript Agent",
    instructions=review_transcript_instructions,
    model="gpt-4o-mini",
    output_type=Transcript,
)


def segment_transcript_instructions(
    ctxWrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    return f"""
Segment the following transcript into coherent chunks and provide a short summary title for each chunk.
Each chunk should be a paragraph around 4 - 7 sentences if possible. 

Transcript:
{ctxWrapper.context}
"""


segment_transcript_agent = Agent(
    name="Segment Transcript Agent",
    instructions=segment_transcript_instructions,
    model="gpt-4o-mini",
    output_type=SegmentedTranscript,
)


class TranscriptProcessor:
    """Processes a Youtube transcript.

    This is not defined as an agent because it is called deterministically from the app.
    """

    def __init__(self, db_path: str = "youtube_videos.db"):
        self.db = VideoDatabase(db_path)

    async def run(self, video_id: str) -> SegmentedTranscript:
        """Process a YouTube video: fetch transcript, segment it, and store results."""
        # Check if video exists in database
        cached_content = self.db.get_video(video_id)
        if cached_content and cached_content["segmented_transcript"]:
            logger.info(f"Retrieved video {video_id} from database")
            return cached_content["segmented_transcript"]

        # If not in database, process the video
        logger.info(f"Processing new video {video_id}")
        transcript = YouTubeTranscriptDownloader.get_transcript(video_id)
        segmented_transcript = await self.process_transcript(transcript)

        # Store in database
        self.db.store_video(
            video_id=video_id,
            url=video_id,
            transcript=transcript,
            segmented_transcript=segmented_transcript,
        )

        return segmented_transcript

    async def process_transcript(self, transcript: str) -> SegmentedTranscript:
        """Process transcript text into segments using LLM."""
        result = await Runner.run(review_transcript_agent, input="", context=transcript)
        logger.info(result)
        result = await Runner.run(
            segment_transcript_agent, input="", context=result.final_output.text
        )
        logger.info(result)

        return result.final_output


if __name__ == "__main__":
    processor = TranscriptProcessor()
    text = """
오늘오전 국립 대전현충원에서 열린 제 10회 서해 수호의 날 기념식 이재명 더불어민주당 대표가 처음으로 기념식에 참석했습니다 선거법 사건 20 무제 판결 뒤 경북 산불피해 원장을 방문했다가 안보 행보에도 나선 겁니다 그러나 냉랭한 분위기도 감지됐습니다 기념 식장을 나서려 했을 때 천남 유족은 이대표 쪽으로 다가서며 거세 하기도 했습니다 폭당 기 상사의 진영 민광기 씨는 어제 SNS 통해 안한 폭침을 부정하고 생존 장면과 유족들에게 막말과 상처를 주고 한마디 반성 없이 서해수호 날 행사를 참석하겠다고 한다며이 대표에게 사과를 요구하기도 했습니다 이대표는 지난 총선에서 이른바 민주당 인사들의 천남 막말논란 때 민주당을 거세게 비판했던 최현일 전 천남 함장과 인사를 나눴습니다 조승 민주당 수석대변인은 남 족들의 같은 입장을 밝혔습니다 그 국가가 결정한 거에 대해서 그 누구도 의심하는 사람 하나도 없습니다 예예 이미 그 우리 대한민국이 대한민국이 또 정부가 어 방 원칙과 방향도 정했고 그 서해 수호에 대한 구등 의지를 규정을 했고 그거에 대해서 당연히 더불어민주당은 또 이재명 대표는 당연히 똑같은 생각 하고 있 기념식에 앞서 민주당은 대전에서 현장 최고위원 회의를 열었습니다으로 시작한 최고위에서 이대표는 북한의 기도발 공격에 맞서 서해바다를 수호한 영웅들을 기억한다고 말문을 열었습니다 제연 해전부터 안한 피격 연평도 전까지 국민의 안전한 일상을 위해 목숨을 쳐 산한 55인의 사들과 모든 장들 에 지금의 대한민국이 있습니다 가슴 개피 경의와 추모의 마음을 전합니다이 대표는 또 최근 서해에 설치된 중국의 구조물을 언급하며 민주당은 모든 영토주권 치매 행위를 단호히 반대하고 서해바다를 더욱 공고하게 지켜낼 것이라고 말했습니다
    """
    segmented_transcript = processor.process_transcript(text)
