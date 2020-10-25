from jamo import hangul_to_jamo

PAD = '_'
EOS = '~'
SPACE = ' '

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + SPACE
symbols = PAD + EOS + VALID_CHARS

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
  sequence = []
  if not 0x1100 <= ord(text[0]) <= 0x1113:
    text = ''.join(list(hangul_to_jamo(text)))
  for s in text:
    sequence.append(_symbol_to_id[s])
  sequence.append(_symbol_to_id['~'])
  return sequence


def sequence_to_text(sequence):
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      result += s
  return result.replace('}{', ' ')
