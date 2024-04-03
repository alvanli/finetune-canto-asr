import re, string

punc = re.compile(u'[\u3002\u3005-\u3007\u3010\u3011\uFF01\uFF0C\uFF1A\uFF1F\uFFE4\uFFE9]')
additional_marks = '、：；？！'
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'

def remove_special_tokens(text):
    text = text.replace("[UNK]", "")
    text = text.replace("[PAD]", "")
    return text


def remove_punc(text):
    text = punc.sub('', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = "".join([c for c in text if c not in additional_marks])
    text = re.sub(chars_to_remove_regex, '', text).lower()
    return text


# import re
# import string

# def remove_punc(text):
#     # Remove punctuation using regex (excluding Chinese characters)
#     text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    
#     # Remove additional marks
#     additional_marks = '、：；？！'
#     text = "".join([c for c in text if c not in additional_marks])
    
#     # Convert to lowercase
#     text = text.lower()
    
#     return text

# # Example usage
# input_text = "Hello, World! 你好，世界！This is a test."
# cleaned_text = remove_punc(input_text)
# print(f"Cleaned text: {cleaned_text}")


def remove_space(text):
    return text.replace(" ", "")


def normalize(text):
    text = remove_special_tokens(text)
    text = remove_punc(text)
    text = remove_space(text)
    return text
