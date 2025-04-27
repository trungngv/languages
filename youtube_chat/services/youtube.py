import re

from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptDownloader:

    @staticmethod
    def extract_video_id(message: str) -> str | None:
        """Extract video ID from a message containing YouTube URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)",
            r"(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]+)",
            r"(?:youtube\.com\/v\/)([a-zA-Z0-9_-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def get_transcript(video_id: str) -> str:
        """Get transcript for a YouTube video."""
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id, languages=["ko"])
            return " ".join([snippet.text for snippet in transcript.snippets])
        except Exception as e:
            raise Exception(f"Failed to get transcript: {str(e)}")


if __name__ == "__main__":
    processor = YouTubeTranscriptDownloader()
    video_id = processor.extract_video_id(
        "here it is https://www.youtube.com/watch?v=RI2DM3as6Wc"
    )
    print(video_id)
    transcript = processor.get_transcript(video_id)
    print(transcript)
