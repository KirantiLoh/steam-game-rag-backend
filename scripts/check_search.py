import json
with open(r"C:\Users\patri\AppData\Local\Temp\search_horror.json") as f:
    d = json.load(f)
print(f'Total results: {d["total"]}')
found = False
for r in d["results"]:
    g = r["game"]
    if "obscure" in g["name"].lower() or "4242750" in str(g["id"]):
        print(f'  FOUND: {g["name"]} (id={g["id"]}) score={r["score"]}')
        found = True
        break
if not found:
    print("  Obscure Horrors NOT found in top results")
    for r in d["results"][:5]:
        g = r["game"]
        print(f'  [{g["id"]}] {g["name"]} - {r["score"]}')
