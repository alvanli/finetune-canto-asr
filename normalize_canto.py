import re, string

punc = re.compile(u'[\u3002\u3005-\u3007\u3010\u3011\uFF01\uFF0C\uFF1A\uFF1F\uFFE4\uFFE9]')
additional_marks = '、：；？！'

def remove_special_tokens(text):
    text = text.replace("[UNK]", "")
    text = text.replace("[PAD]", "")
    return text


def remove_punc(text):
    text = punc.sub('', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = "".join([c for c in text if c not in additional_marks])
    return text


def remove_space(text):
    return text.replace(" ", "")


def normalize(text):
    text = remove_special_tokens(text)
    text = remove_punc(text)
    text = remove_space(text)
    return text
