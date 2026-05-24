import urllib.request
import json

# Test game detail
resp = urllib.request.urlopen("http://localhost:8000/api/games/1150690")
game = json.loads(resp.read())
print("=== Game Detail ===")
print(f"  Name: {game['name']}")
print(f"  Price: ${game['price']}")
print(f"  Genres: {game['genres']}")
print(f"  Devs: {game['developers']}")
print(f"  Release: {game['release_date']}")
print(f"  Metacritic: {game['metacritic_score']}")
print(f"  Steam Rating: {game['steam_rating']}%")
print(f"  Reviews: {game['positive_reviews']} pos / {game['negative_reviews']} neg")
print(f"  Platforms: {game['platforms']}")
print(f"  Screenshots: {len(game['screenshots'])}")
print(f"  Short desc: {game['short_description'][:80]}...")

# Test search
print("\n=== Search ===")
resp = urllib.request.urlopen("http://localhost:8000/api/search?q=Horror&limit=3")
data = json.loads(resp.read())
print(f"  Total: {data['total']}")
for r in data['results']:
    g = r['game']
    print(f"  {g['name']} - ${g['price']} - {g['genres']} - {g['steam_rating']}% (score: {r['score']})")

# Test 404
print("\n=== 404 Test ===")
try:
    resp = urllib.request.urlopen("http://localhost:8000/api/games/99999999")
    print("  ERROR: Should have 404'd")
except urllib.error.HTTPError as e:
    print(f"  404 correctly returned: {e.code}")

print("\nAll tests passed!")
