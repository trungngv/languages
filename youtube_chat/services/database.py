import sqlite3

from youtube_chat.pydantic_models import SegmentedTranscript


class VideoDatabase:
    def __init__(self, db_path: str = "youtube_videos.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with the required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    transcript TEXT,
                    segmented_transcript TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def store_video(
        self,
        video_id: str,
        transcript: str,
        segmented_transcript: SegmentedTranscript,
    ):
        """Store video content in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO videos (video_id, transcript, segmented_transcript)
                VALUES (?, ?, ?)
            """,
                (
                    video_id,
                    transcript,
                    segmented_transcript.model_dump_json(),
                ),
            )
            conn.commit()

    def get_video(self, video_id: str) -> dict[str, any]:
        """Retrieve video content from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT transcript, segmented_transcript
                FROM videos
                WHERE video_id = ?
            """,
                (video_id,),
            )
            result = cursor.fetchone()

            if result:
                transcript, segmented_transcript = result
                return {
                    "video_id": video_id,
                    "transcript": transcript,
                    "segmented_transcript": (
                        SegmentedTranscript.model_validate_json(segmented_transcript)
                        if segmented_transcript
                        else None
                    ),
                }

            return {}
