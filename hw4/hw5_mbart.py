"""
In this code, we implemented unsupervised neural machine translation using multilingual pretrained language model mBart
"""

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm

SOS_token = "<SOS>"
EOS_token = "<EOS>"


def clean(strx):
    """
  input: string with bpe, EOS
  output: list without bpe, EOS
  """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


# Load input data
input_path = "data/fren.test.bpe"
input_data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for x in f:
        temp = x.strip().split("|||")
        input_data.append(clean(temp[0]))

# Load Pretrained language model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Set En-Fr Translation Model Setting
output = []
tokenizer.src_lang = "fr_XX"
for x in tqdm(input_data):
    x = clean(x)
    encoded_fr = tokenizer(x, return_tensors="pt")
    generated_tokens = model.generate(**encoded_fr, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    output.append(translation[0])

# Write to file
with open("translations", 'w', encoding="utf-8") as f:
    for x in output:
        temp = x + "\n"
        f.write(temp)
