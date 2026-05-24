"""Test script to validate reranker optimization changes."""
from reranker import Reranker


def test_prepare_text():
    """Test the smart text preparation method."""
    reranker = Reranker()

    # Test 1: Normal game with all fields
    print("=== Test 1: Normal game with all fields ===")
    name = "The Witcher 3: Wild Hunt"
    short = "As war rages on throughout the Northern Realms, you take on the greatest contract of your life — tracking down the Child of Prophecy, a living weapon that can alter the shape of the world."
    detailed = "The Witcher 3: Wild Hunt is a story-driven open world RPG set in a visually stunning fantasy universe full of meaningful choices and impactful consequences. In The Witcher you play as Geralt of Rivia, one of a dying caste of monster hunters, and embark on an epic journey in a war-ravaged world that will inevitably lead you to confront a foe darker than anything humanity has faced so far—the Wild Hunt." * 3  # Make it longer

    result = reranker.prepare_text(name, short, detailed)
    print(f"Input lengths: name={len(name)}, short={len(short)}, detailed={len(detailed)}")
    print(f"Output length: {len(result)} chars (~{len(result)//4} tokens)")
    print(f"Preview: {result[:200]}...")
    assert len(result) < 2500, "Text should fit within ~600 token budget"
    assert name in result, "Name should be included"
    assert short in result, "Short description should be fully included"
    print("✓ Test 1 passed\n")

    # Test 2: Very long detailed description
    print("=== Test 2: Very long detailed description (>4000 chars) ===")
    name = "Cyberpunk 2077"
    short = "Cyberpunk 2077 is an open-world, action-adventure RPG set in the dark future of Night City."
    detailed = "Welcome to Night City, a megalopolis obsessed with power, glamor, and ceaseless body modification. " * 100  # ~10,000 chars

    result = reranker.prepare_text(name, short, detailed)
    print(f"Input lengths: name={len(name)}, short={len(short)}, detailed={len(detailed)}")
    print(f"Output length: {len(result)} chars (~{len(result)//4} tokens)")
    assert len(result) < 2500, "Text should be truncated to fit budget"
    assert name in result, "Name should be included"
    assert short in result, "Short description should be fully included"
    print("✓ Test 2 passed\n")

    # Test 3: Missing short description
    print("=== Test 3: Missing short description ===")
    name = "Indie Puzzle Game"
    short = ""
    detailed = "This is an innovative puzzle game that challenges your mind with unique mechanics and beautiful visuals. " * 20

    result = reranker.prepare_text(name, short, detailed)
    print(f"Input lengths: name={len(name)}, short={len(short)}, detailed={len(detailed)}")
    print(f"Output length: {len(result)} chars (~{len(result)//4} tokens)")
    assert len(result) < 2500, "Text should fit within budget"
    assert name in result, "Name should be included"
    print("✓ Test 3 passed\n")

    # Test 4: Very long name (edge case)
    print("=== Test 4: Very long name (edge case) ===")
    name = "Super Ultra Mega Hyper Extreme Ultimate Deluxe Premium Special Edition Game of the Year Complete Collection Remastered" * 2
    short = "A game with an absurdly long name."
    detailed = "This game has a really long name but otherwise normal content."

    result = reranker.prepare_text(name, short, detailed)
    print(f"Input lengths: name={len(name)}, short={len(short)}, detailed={len(detailed)}")
    print(f"Output length: {len(result)} chars (~{len(result)//4} tokens)")
    assert len(result) < 2500, "Text should be truncated to fit budget"
    assert result.startswith(name[:197]), "Name should be truncated if too long"
    print("✓ Test 4 passed\n")

    # Test 5: All empty fields
    print("=== Test 5: All empty fields ===")
    result = reranker.prepare_text("", "", "")
    print(f"Output length: {len(result)} chars")
    assert len(result) == 0, "Empty input should produce empty output"
    print("✓ Test 5 passed\n")

    print("=" * 50)
    print("All tests passed! ✓")


def test_batch_size():
    """Test that batch size is increased."""
    print("\n=== Testing batch size configuration ===")
    reranker = Reranker()
    assert reranker.batch_size == 64, f"Expected batch_size=64, got {reranker.batch_size}"
    print(f"✓ Batch size correctly set to {reranker.batch_size}")


def test_rerank_with_prepared_text():
    """Test reranking with prepared text."""
    print("\n=== Testing rerank with prepared text ===")
    reranker = Reranker()

    query = "horror survival game"

    # Simulate candidates with pre-prepared text
    candidates = [
        ("12345", reranker.prepare_text(
            "Resident Evil Village",
            "Experience survival horror like never before in the eighth major installment in the Resident Evil franchise.",
            "Set a few years after the horrifying events in the critically acclaimed Resident Evil 7 biohazard..." * 10
        )),
        ("67890", reranker.prepare_text(
            "The Last of Us",
            "A third person survival action game.",
            "The Last of Us is a genre-defining experience blending survival and action elements..." * 10
        )),
        ("11111", reranker.prepare_text(
            "Tetris",
            "Classic puzzle game.",
            "Stack blocks to clear lines." * 10
        )),
    ]

    results = reranker.rerank(query, candidates, k=3)

    print(f"Query: '{query}'")
    print(f"Results: {len(results)} games ranked")
    for i, (app_id, score) in enumerate(results, 1):
        print(f"  {i}. App ID: {app_id}, Score: {score:.4f}")

    # Horror games should rank higher than Tetris
    tetris_score = next(score for app_id, score in results if app_id == "11111")
    re_score = next(score for app_id, score in results if app_id == "12345")
    assert re_score > tetris_score, "Resident Evil should rank higher than Tetris for horror query"

    print("✓ Reranking works correctly with prepared text\n")


if __name__ == "__main__":
    test_prepare_text()
    test_batch_size()
    test_rerank_with_prepared_text()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓✓✓")
    print("=" * 50)
