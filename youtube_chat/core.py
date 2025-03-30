from pydantic import BaseModel, Field


class Transcript(BaseModel):
    text: str
    language: str


class Segment(BaseModel):
    index: int
    text: str = Field(description="The text of the segment")
    summary: str = Field(description="A one sentence summary of the segment")


class SegmentedTranscript(BaseModel):
    segments: list[Segment]

    def get_summaries(self) -> str:
        return "\n".join(
            [f"Segment {ix + 1}: {s.summary}" for ix, s in enumerate(self.segments)]
        )


class LearningItem(BaseModel):
    item: str
    meaning: str


class Sentence(BaseModel):
    sentence_number: int = Field(description="Index of the sentence in the segment")
    original: str = Field(description="The original sentence")
    translation: str = Field(description="English translation")
    vocabulary: list[LearningItem] = Field(description="a list of vocabulary")
    expressions: list[LearningItem] = Field(description="a list of idiom expressions")
    grammar_points: list[LearningItem] = Field(description="a list of grammar points")


class SegmentExplanation(BaseModel):
    sentences: list[Sentence]

    def get_sentences(self) -> str:
        result = "## Sentence-by-Sentence Explanation\n"
        for sentence in self.sentences:
            result += self._format_sentence(sentence) + "\n"

        return result

    def _format_sentence(self, sentence: Sentence) -> str:
        """Format a single sentence with its analysis."""
        result = f"**Sentence**: {sentence.original}\n"
        result += f"{sentence.translation}\n"
        result += self._format_items(sentence.vocabulary, "Vocabulary")
        if sentence.expressions:
            result += self._format_items(sentence.expressions, "Expressions")
        if sentence.grammar_points:
            result += self._format_items(sentence.grammar_points, "Grammar Points")

        return result

    def _format_items(self, items: list[LearningItem], heading: str) -> str:
        """Format list of learning items into markdown."""
        result = f"**{heading}**\n"
        result += "\n".join([f"- **{item.item}**: {item.meaning}\n" for item in items])

        return result
