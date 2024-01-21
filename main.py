import asyncio
from solver import CaptchaSolver


async def main():
    solver = CaptchaSolver(
        ru_model="models/moderuimp.onnx",
        en_model="models/modelen.onnx",
    )
    result = await solver.read("captcha.png")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
