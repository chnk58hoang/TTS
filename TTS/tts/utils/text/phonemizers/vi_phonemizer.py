from typing import Dict

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.characters import _punctuations
from TTS.tts.utils.text.vietnamese.phonemizer import vietnamese_text_to_phonemes
from g2p_en import G2p


class ViPhonemizer(BasePhonemizer):
    """
    TTS Vietnamese phonemizer using functions in `TTS.tts.utils.text.vietnamese.phonemizer`
    """

    language = "vi"

    def __init__(self,
                 punctuations=_punctuations,
                 keep_puncs=True,
                 **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language,
                         punctuations=punctuations,
                         keep_puncs=keep_puncs)
        self.g2p_model = G2p()


    @staticmethod
    def name():
        return "vi_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        ph = vietnamese_text_to_phonemes(text, self.g2p_model)
        return ph

    def phonemize(self, text: str, separator="|", language=None) -> str:
        """Custom phonemize for Vietnamese
        Skip pre-post processing steps used by the other phonemizers.
        """
        return self._phonemize(text, separator)

    @staticmethod
    def supported_languages():
        return {"vi": "Vietnamese"}

    def version(self) -> str:   
        return "0.0.1"

    def is_available(self) -> bool:
        return True


