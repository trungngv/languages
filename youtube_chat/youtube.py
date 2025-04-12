import re

from youtube_transcript_api import YouTubeTranscriptApi


def extract_youtube_url(message: str) -> str | None:
    """Extract YouTube URL from message."""
    url_pattern = (
        r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]+"
    )
    match = re.search(url_pattern, message)
    return match.group(0) if match else None


class YouTubeTranscriptDownloader:
    def __init__(self):
        pass

    def extract_video_id(self, url: str) -> str | None:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)",
            r"(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]+)",
            r"(?:youtube\.com\/v\/)([a-zA-Z0-9_-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_transcript(self, url: str) -> str:
        """Get transcript for a YouTube video."""
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        try:
            transcript = YouTubeTranscriptApi().fetch(video_id, languages=["ko"])
            return " ".join([snippet.text for snippet in transcript.snippets])
        except Exception as e:
            raise Exception(f"Failed to get transcript: {str(e)}")


if __name__ == "__main__":
    processor = YouTubeTranscriptDownloader()
    transcript = processor.get_transcript("https://www.youtube.com/watch?v=RI2DM3as6Wc")
    print(transcript)
