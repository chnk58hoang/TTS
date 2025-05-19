from pyvi import ViTokenizer
import json


dict_phoneme = json.load(open('dict_phoneme.json', 'r'))
all_syllable = list(dict_phoneme.keys())


def vietnamese_text_to_phonemes(text: str,
                                g2p_model) -> str:
    """Convert Vietnamese text to phonemes using a dictionary.

    Args:
        text (str): Input Vietnamese text.

    Returns:
        str: Phonemized text.
    """
    text = text.lower()
    segment_text = ViTokenizer.tokenize(text)
    segment_text = segment_text.split()
    convert_text = []
    for sentence in [segment_text]:
        convert_sentence = []
        for word in sentence:
            if '_' in word:
                for sub_word in word.replace('_', ' ').split():
                    convert_sentence.append(sub_word)
            else:
                convert_sentence.append(word)
        sub_cv = []
        for sym in convert_sentence:
            try:
                if sym in all_syllable:
                    for ph in dict_phoneme[sym].split():
                        sub_cv.append(ph)
                else:
                    phone_eng = g2p_model(sym)
                    for ph in phone_eng:
                        sub_cv.append(ph)
            except:
                sub_cv.append(sym)
    convert_text.append(" ".join(sub_cv))
    convert_text = ' '.join(convert_text)
    return convert_text
