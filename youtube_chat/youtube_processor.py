from youtube_transcript_api import YouTubeTranscriptApi

from youtube_chat.core import Transcript


class YouTubeProcessor:
    def __init__(self):
        self.api = YouTubeTranscriptApi()

    def get_transcript(self, url: str, language_code: str = "ko") -> Transcript:
        """Extract transcript from a YouTube video."""
        try:
            video_id = self._extract_video_id(url)
            transcript_data = self.api.fetch(video_id, languages=[language_code])

            text = " ".join(item.text for item in transcript_data.snippets)
            return Transcript(text=text, language=language_code)
        except Exception as e:
            raise Exception(f"Error extracting transcript: {str(e)}")

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        import re

        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)",
            r"youtube\.com\/embed\/([^&\n?]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise ValueError("Invalid YouTube URL")


if __name__ == "__main__":
    processor = YouTubeProcessor()
    transcript = processor.get_transcript("https://www.youtube.com/watch?v=RI2DM3as6Wc")
    print(transcript)
