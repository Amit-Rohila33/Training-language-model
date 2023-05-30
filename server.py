import random
from fastapi import FastAPI
from pydantic import BaseModel


class InputData(BaseModel):
    text: str


class GeneratedText(BaseModel):
    generated_text: str


app = FastAPI()


@app.post("/generate", response_model=GeneratedText)
async def generate_text(data: InputData):
    # Complex logic: Generate a random sequence of characters based on input text length
    generated_text = ""

    for char in data.text:
        if char.isalpha():
            # Randomly decide whether to append the uppercase or lowercase version of the character
            if random.choice([True, False]):
                generated_text += char.upper()
            else:
                generated_text += char.lower()
        else:
            generated_text += char

    return {"generated_text": generated_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
