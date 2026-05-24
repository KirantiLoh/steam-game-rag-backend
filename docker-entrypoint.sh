#!/bin/bash
set -e

echo "Waiting for Elasticsearch..."
until curl -s "$ES_URL" > /dev/null 2>&1; do
  sleep 2
done
echo "Elasticsearch is ready"

INDEX_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" "$ES_URL/steam_games/_count" 2>/dev/null || echo "000")
DOC_COUNT=$(curl -s "$ES_URL/steam_games/_count" 2>/dev/null | python -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo "0")

if [ "$DOC_COUNT" -lt "1000" ]; then
  echo "Index steam_games empty ($DOC_COUNT docs). Running indexer..."
  python scripts/elastic_index.py
  echo "Indexing complete"
else
  echo "Index steam_games has $DOC_COUNT docs — skipping indexing"
fi

echo "Starting uvicorn..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
