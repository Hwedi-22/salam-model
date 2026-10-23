"""الدالة اللي الباك إند هيستدعيها.

    from predict import predict
    predict("نص عربي")      # بيستخدم موديل العربي
    predict("english text")  # بيستخدم موديل الإنجليزي

مهم: الدالة دي بتاخد النص وترجع نتيجة فقط، ومبتكتبش النص في أي لوج أو ملف.
"""
import os
import re
import joblib

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIRS = {"ar": "outputs_ar", "en": "outputs_en"}
_bundles = {}

DISCLAIMER = (
    "هذه النتيجة تعليمية للتوعية فقط، وليست تشخيصاً طبياً. "
    "إذا كنت تشعر بضيق مستمر، يُنصح بالتحدث مع مختص."
)
MIN_CHARS = 5

_AR = re.compile(r"[؀-ۿ]")
_EN = re.compile(r"[A-Za-z]")


def detect_language(text: str):
    ar, en = len(_AR.findall(text)), len(_EN.findall(text))
    if ar == 0 and en == 0:
        return None
    return "ar" if ar >= en else "en"


def _load(lang):
    if lang not in _bundles:
        path = os.path.join(_BASE, _MODEL_DIRS[lang], "baseline_model.joblib")
        _bundles[lang] = joblib.load(path)
    return _bundles[lang]


def predict(text: str) -> dict:
    if not isinstance(text, str) or len(text.strip()) < MIN_CHARS:
        return {"status": "invalid_input", "message": "النص قصير جداً أو فارغ.", "disclaimer": DISCLAIMER}

    lang = detect_language(text)
    if lang is None:
        return {"status": "unsupported_language", "message": "النص لا يحتوي على حروف عربية أو إنجليزية.",
                "disclaimer": DISCLAIMER}

    b = _load(lang)
    meta = b["meta"]
    p = b["pipeline"].predict_proba([text])[0]
    classes = list(b["pipeline"].classes_)
    score = float(p[classes.index(1)])
    # ملحوظة: "no_strong_indicators" لا تعني أن المستخدم بخير.
    indicator = "indicators_detected" if score >= 0.5 else "no_strong_indicators"
    return {
        "status": "ok",
        "language": lang,
        "indicator": indicator,
        "score": round(score, 3),
        "model_version": f"{meta['model_version']}-{lang}",
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    print(predict("حاسس إني تعبان قوي ومش عايز أكلم حد"))
    print(predict("النهارده يوم جميل وخارج مع صحابي"))
    print(predict("I feel so empty and tired of everything"))
    print(predict("12345 !!!"))
