"""
Agent for processing transcripts. This can be split into more specialised agents later.
"""

import logging

from llm_solution.llms import OpenAIClient
from services.database import VideoDatabase
from services.youtube import YouTubeTranscriptDownloader

from youtube_chat.pydantic_models import (
    Segment,
    SegmentedTranscript,
    SegmentExplanation,
    Transcript,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, llm_client: OpenAIClient, db_path: str = "youtube_videos.db"):
        self.llm_client = llm_client
        self.transcript_downloader = YouTubeTranscriptDownloader()
        self.db = VideoDatabase(db_path)

    def process(self, video_url: str) -> SegmentedTranscript:
        """Process a YouTube video: fetch transcript, segment it, and store results."""
        video_id = self.transcript_downloader.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        # Check if video exists in database
        cached_content = self.db.get_video(video_id)
        if cached_content and cached_content["segmented_transcript"]:
            logger.info(f"Retrieved video {video_id} from database")
            return cached_content["segmented_transcript"]

        # If not in database, process the video
        logger.info(f"Processing new video {video_id}")
        transcript = self.transcript_downloader.get_transcript(video_url)
        segmented_transcript = self.process_transcript(transcript)

        # Store in database
        self.db.store_video(
            video_id=video_id,
            url=video_url,
            transcript=transcript,
            segmented_transcript=segmented_transcript,
        )

        return segmented_transcript

    def process_transcript(self, transcript: str) -> SegmentedTranscript:
        """Process transcript text into segments using LLM."""
        reviewed_transcript = self._review_and_format_transcript(transcript)
        segmented_transcript = self._segment_transcript(reviewed_transcript.text)

        return segmented_transcript

    def _review_and_format_transcript(self, text: str) -> Transcript:
        """Review the transcript to determine language and add punctuation if needed."""
        prompt = f"""
Review the following transcript and:
1. Determine the language
2. Add proper punctuation if missing
3. Format the text for better readability

Transcript:
{text}

Provide the response in the following JSON format:
{{
    "language": "language code e.g. 'ko' for Korean, 'en' for English, etc.",
    "text": "text with proper punctuation",
}}
"""
        response = self.llm_client.call(
            system_prompt=prompt,
            user_prompt=text,
            output_model=Transcript,
        )

        return response

    def _segment_transcript(self, text: str) -> SegmentedTranscript:
        """Segment the transcript into meaningful chunks with summaries."""

        prompt = f"""
Segment the following text into meaningful chunks and provide a summary for each.
Each chunk should be a paragraph around 4 - 7 sentences (not a strict rule).

Text:
{text}
"""
        response = self.llm_client.call(
            system_prompt=prompt,
            user_prompt=text,
            output_model=SegmentedTranscript,
        )

        return response

    def get_segment_details(self, segment: Segment) -> SegmentExplanation:
        """Get detailed information about a segment including translations and explanations."""

        prompt = f"""
Analyze the following text segment sentence by sentence. For each sentence:
1. Provide English translation
2. List key vocabulary words with their meanings 
3. Identify common expressions or idioms with explanations
4. Note 1 -2 important grammar points based on the difficulty level of the text (e.g. if the text is advanced, note more complex grammar points)

Text:
{segment.text}

Provide the response in the specified format.
"""
        response = self.llm_client.call(
            system_prompt="You are a helpful language teacher explaining the text sentence by sentence.",
            user_prompt=prompt,
            output_model=SegmentExplanation,
        )

        return response


if __name__ == "__main__":
    processor = VideoProcessor(OpenAIClient())
    text = """
오늘오전 국립 대전현충원에서 열린 제 10회 서해 수호의 날 기념식 이재명 더불어민주당 대표가 처음으로 기념식에 참석했습니다 선거법 사건 20 무제 판결 뒤 경북 산불피해 원장을 방문했다가 안보 행보에도 나선 겁니다 그러나 냉랭한 분위기도 감지됐습니다 기념 식장을 나서려 했을 때 천남 유족은 이대표 쪽으로 다가서며 거세 하기도 했습니다 폭당 기 상사의 진영 민광기 씨는 어제 SNS 통해 안한 폭침을 부정하고 생존 장면과 유족들에게 막말과 상처를 주고 한마디 반성 없이 서해수호 날 행사를 참석하겠다고 한다며이 대표에게 사과를 요구하기도 했습니다 이대표는 지난 총선에서 이른바 민주당 인사들의 천남 막말논란 때 민주당을 거세게 비판했던 최현일 전 천남 함장과 인사를 나눴습니다 조승 민주당 수석대변인은 남 족들의 같은 입장을 밝혔습니다 그 국가가 결정한 거에 대해서 그 누구도 의심하는 사람 하나도 없습니다 예예 이미 그 우리 대한민국이 대한민국이 또 정부가 어 방 원칙과 방향도 정했고 그 서해 수호에 대한 구등 의지를 규정을 했고 그거에 대해서 당연히 더불어민주당은 또 이재명 대표는 당연히 똑같은 생각 하고 있 기념식에 앞서 민주당은 대전에서 현장 최고위원 회의를 열었습니다으로 시작한 최고위에서 이대표는 북한의 기도발 공격에 맞서 서해바다를 수호한 영웅들을 기억한다고 말문을 열었습니다 제연 해전부터 안한 피격 연평도 전까지 국민의 안전한 일상을 위해 목숨을 쳐 산한 55인의 사들과 모든 장들 에 지금의 대한민국이 있습니다 가슴 개피 경의와 추모의 마음을 전합니다이 대표는 또 최근 서해에 설치된 중국의 구조물을 언급하며 민주당은 모든 영토주권 치매 행위를 단호히 반대하고 서해바다를 더욱 공고하게 지켜낼 것이라고 말했습니다
    """
    segmented_transcript = processor.process_transcript(text)
    segment_explanation = processor.get_segment_details(
        segmented_transcript.segments[0]
    )
    print(segment_explanation.get_sentences())
