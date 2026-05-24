"""Manual testing script for chat API with Gemini."""
import asyncio
import httpx
import sys


async def test_streaming():
    """Test streaming response."""
    print("=" * 60)
    print("Testing Streaming Chat (SSE)")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/api/chat",
            json={
                "query": "I want dark souls-like games but easier",
                "stream": True
            },
            timeout=60.0
        ) as response:
            session_id = None

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("event: session"):
                    continue
                elif line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    if session_id is None and len(data) == 36:  # UUID
                        session_id = data
                        print(f"\nSession: {session_id}\n")
                    elif data.isdigit():
                        print(f"[Retrieved {data} games]\n")
                    elif data == "success":
                        print("\n\n[Stream complete]")
                    else:
                        # Unescape newlines
                        data = data.replace('\\n', '\n')
                        print(data, end="", flush=True)
                elif line.startswith("event: done"):
                    break
                elif line.startswith("event: error"):
                    print(f"\n\n[Error occurred]")

    print("\n" + "=" * 60)


async def test_non_streaming():
    """Test non-streaming response."""
    print("\n" + "=" * 60)
    print("Testing Non-Streaming Chat")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={
                "query": "Recommend cozy games under $20",
                "stream": False,
                "price_max": 20
            },
            timeout=60.0
        )

        data = response.json()
        print(f"\nSession: {data['session_id']}")
        print(f"Games Retrieved: {data['games_retrieved']}")
        print(f"\nResponse:\n{data['response']}")

    print("=" * 60)


async def test_conversation():
    """Test multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Testing Multi-Turn Conversation")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Turn 1
        print("\n[Turn 1: Initial query]")
        resp1 = await client.post(
            "http://localhost:8000/api/chat",
            json={"query": "Recommend strategy games", "stream": False},
            timeout=60.0
        )
        data1 = resp1.json()
        session_id = data1["session_id"]
        print(f"Response: {data1['response'][:200]}...")

        # Turn 2
        print("\n[Turn 2: Follow-up]")
        resp2 = await client.post(
            "http://localhost:8000/api/chat",
            json={
                "query": "What about turn-based ones?",
                "session_id": session_id,
                "stream": False
            },
            timeout=60.0
        )
        data2 = resp2.json()
        print(f"Response: {data2['response'][:200]}...")

        # Turn 3
        print("\n[Turn 3: Specific question]")
        resp3 = await client.post(
            "http://localhost:8000/api/chat",
            json={
                "query": "Tell me more about the first game you mentioned",
                "session_id": session_id,
                "stream": False
            },
            timeout=60.0
        )
        data3 = resp3.json()
        print(f"Response: {data3['response'][:200]}...")

    print("=" * 60)


async def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("Testing Health Check")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/chat/health")
        data = response.json()

        print(f"\nStatus: {data['status']}")
        print(f"Provider: {data['provider']}")
        print(f"Model: {data['model']}")
        print(f"Free Tier: {data['free_tier']}")
        print(f"Limits: {data['limits']}")
        print(f"RAG Config: {data['rag_config']}")

    print("=" * 60)


async def interactive_mode():
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("Type 'quit' to exit, 'new' to start new session")
    print("=" * 60)

    session_id = None

    async with httpx.AsyncClient() as client:
        while True:
            try:
                query = input("\nYou: ").strip()

                if not query:
                    continue

                if query.lower() == 'quit':
                    print("Goodbye!")
                    break

                if query.lower() == 'new':
                    session_id = None
                    print("[New session started]")
                    continue

                # Stream response
                print("Assistant: ", end="", flush=True)

                async with client.stream(
                    "POST",
                    "http://localhost:8000/api/chat",
                    json={
                        "query": query,
                        "stream": True,
                        "session_id": session_id
                    },
                    timeout=60.0
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        if line.startswith("event: session"):
                            continue
                        elif line.startswith("data: "):
                            data = line[6:]

                            if session_id is None and len(data) == 36:
                                session_id = data
                            elif data == "success":
                                print()
                            elif not data.isdigit():
                                data = data.replace('\\n', '\n')
                                print(data, end="", flush=True)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n\nError: {e}")


async def main():
    """Run all manual tests or interactive mode."""
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await interactive_mode()
    else:
        try:
            await test_health()
            await test_streaming()
            await test_non_streaming()
            await test_conversation()

            print("\n✅ All manual tests completed successfully!")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\nUsage:")
    print("  python test_chat_manual.py              # Run all tests")
    print("  python test_chat_manual.py interactive  # Interactive chat mode")
    print()

    asyncio.run(main())
