import re


class CitationParser:
    """
    Extracts individual claims together with their citations.

    Example:

    Employees receive 24 leaves [1].
    Remote work is allowed [2][3].

    ->
    [
        ("Employees receive 24 leaves", [1]),
        ("Remote work is allowed", [2,3])
    ]

    Each line (bullet, table row, paragraph line) is split into sentences.
    Citation markers written after the closing punctuation ("... leave. [1]")
    belong to the sentence they follow.
    """

    citation_pattern = re.compile(r"\[(\d+)\]")
    _sentence_end = re.compile(r"(?<=[.!?])\s+")
    _space_before_punctuation = re.compile(r"\s+([.,;:!?])")

    def parse(
        self,
        answer: str,
    ) -> list[tuple[str, list[int]]]:

        claims: list[tuple[str, list[int]]] = []

        for line in answer.splitlines():
            previous_in_line = False

            for sentence in self._sentence_end.split(line.strip()):
                citations = [int(c) for c in self.citation_pattern.findall(sentence)]
                clean_sentence = self.citation_pattern.sub("", sentence).strip()
                clean_sentence = self._space_before_punctuation.sub(r"\1", clean_sentence)

                if clean_sentence.strip("*_`-|> ") == "":
                    # only markers (or markdown punctuation): they cite the
                    # sentence just before them on the same line
                    if citations and previous_in_line:
                        text, existing = claims[-1]
                        claims[-1] = (text, existing + citations)
                    continue

                claims.append((clean_sentence, citations))
                previous_in_line = True

        return claims
