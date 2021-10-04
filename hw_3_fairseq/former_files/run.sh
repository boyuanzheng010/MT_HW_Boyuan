echo -e "Die maschinelle Übersetzung ist schwer zu kontrollieren.\thard\ttoinfluence" \
| normalize.py | tok.py \
| fairseq-interactive "model/wmt19.de-en.ffn8192/" \
  --path "model/wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt" \
  --bpe fastbpe \
  --bpe-codes "model/wmt19.de-en.ffn8192/ende30k.fastbpe.code" \
  --constraints \
  -s de -t en \
  --beam 10