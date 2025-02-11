import asyncio
from book_model import reload_model, get_recommendations

async def test_reload_model():
    print("Testing reload_model...")
    try:
        await reload_model()
        print("reload_model executed successfully.")
    except Exception as e:
        print(f"Error in reload_model: {e}")

async def test_get_recommendations(book: str, isbn: bool = False):
    print(f"Testing get_recommendations for book: {book} (isbn={isbn})")
    try:
        recommendations = await get_recommendations(book, isbn)
        if isinstance(recommendations, list):
            print("Recommendations:")
            for rec in recommendations:
                print(f"Title: {rec['title']}, Similarity: {rec['similarity']}")
        else:
            print(f"No recommendations found: {recommendations}")
    except Exception as e:
        print(f"Error in get_recommendations: {e}")

async def main():
    # Test reload_model
    await test_reload_model()

    # Test get_recommendations with a book title
    await test_get_recommendations("Where the Heart Is (Oprah's Book Club (Paperback))", isbn=False)

    # Test get_recommendations with an ISBN
    await test_get_recommendations("034545104X", isbn=True)

    # Test get_recommendations with a non-existent book
    await test_get_recommendations("Non-Existent Book", isbn=False)

if __name__ == "__main__":
    asyncio.run(main())
