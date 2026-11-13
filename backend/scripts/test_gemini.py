
from google import genai
from app.core.config import settings


def main():
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    client = genai.Client(
        api_key=settings.gemini_api_key
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=(
            "You are the language model behind InsightOps AI. "
            "Explain in two sentences what a business KPI is."
        )
    )

    print(response.text)


if __name__ == "__main__":
    main()
