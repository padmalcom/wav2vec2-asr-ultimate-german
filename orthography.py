from dataclasses import dataclass

@dataclass
class Orthography:
	do_lower_case = False
	vocab_file = None
	word_delimiter_token = "|"
	translation_table = {}
	words_to_remove = set()
	untransliterator = None
	tokenizer = None