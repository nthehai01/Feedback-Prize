import re

SPAN_TYPE_TO_PATTERN = {
    'word': r'\s+',
    "paragraph": r' *[\n][\'"\)\]]* *',
    "sentence": r' *[\.\?!\n][\'"\)\]]* *',
}


def get_span_from_text(sequence, span_type):
    def split_sequence_by_pattern(sequence, pattern):
        """ Split a sequence by a pattern.

        Args:
            sequence (str): A sequence.
            pattern (str): A pattern.

        Returns:
            A list of substrings.
        """
        return re.split(pattern, sequence)
    
    spans = split_sequence_by_pattern(sequence, SPAN_TYPE_TO_PATTERN[span_type])
    
    spans = [x for x in spans if x!=""]
    
    span_offset = []
    start = 0
    for w in spans:
        r = sequence[start:].find(w)

        if r==-1:
            raise NotImplementedError
        else:
            start = start+r
            end   = start+len(w)
            span_offset.append((start,end))
        start = end

    return spans,span_offset
