"""تنظيف النص العربي - مشروع SALAM.

مهم: مش بنشيل كلمات النفي (مش / لا / ما) لأن معنى الجملة بيتغير بدونها.
"""
import re

_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")   # التشكيل
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")                           # حرف متكرر 3 مرات أو أكتر
_PUNCT = re.compile(r"[\.\,\!\?\;\:\"'\(\)\[\]\{\}\-_\*/\\|~`^<>=+%$&#،؛؟…«»]")
_SPACES = re.compile(r"\s+")
_BIDI = re.compile(r"[‎‏‪-‮⁦-⁩]")


def normalize_arabic(text) -> str:
    text = "" if text is None else str(text)
    text = text.replace("_x000D_", " ")          # بقايا الإكسل
    text = _BIDI.sub("", text)                    # علامات اتجاه النص المخفية
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _DIACRITICS.sub("", text)
    text = text.replace("\u0640", "")            # التطويل ـ
    text = re.sub("[إأآٱ]", "ا", text)           # توحيد الألف
    text = text.replace("ى", "ي")                # الألف المقصورة -> ياء
    text = text.replace("ة", "ه")                # التاء المربوطة -> هاء
    text = _REPEAT.sub(r"\1\1", text)            # وحشتنييييي -> وحشتنيي
    text = _PUNCT.sub(" ", text)
    text = text.lower()                          # للحروف اللاتينية لو موجودة
    text = _SPACES.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    sample = "أنا تعبااااان جدًا ومش عايز أكلم حد!!! http://x.com @user"
    print(normalize_arabic(sample))
