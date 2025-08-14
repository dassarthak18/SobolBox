import gzip, pickle, os

MEMO_FILE = "memo.pkl.gz"

if os.path.exists(MEMO_FILE):
  with gzip.open(MEMO_FILE, "rb") as f:
    memo = pickle.load(f)
  print("Loaded memo cache.")
else:
  memo = {}

def save_memo():
  with gzip.open(MEMO_FILE, "wb") as f:
    pickle.dump(memo, f)
  print("Saved memo cache.")

def clear_memo():
  memo.clear()
  print("Cleared memo cache.")
