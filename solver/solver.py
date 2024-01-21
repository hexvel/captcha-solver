from typing import Any, Coroutine, Literal
import cv2

import numpy as np
import onnxruntime as onr

from PIL import Image

from config import Chars


class CaptchaSolver:
    def __init__(
        self,
        ru_model: str | None = None,
        en_model: str | None = None,
    ) -> None:
        self.img_width = 130
        self.img_height = 50
        self.max_length = 8

        self.ModelRu = onr.InferenceSession(ru_model)
        self.ModelName = self.ModelRu.get_inputs()[0].name

        self.ModelEn = onr.InferenceSession(en_model)
        self.ModelName = self.ModelEn.get_inputs()[0].name

    async def extend_ru(self, img) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = img.transpose([2, 1, 0])
        img = np.expand_dims(img, axis=0)

        return img

    async def extend_en(self, img) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        img = img.transpose([1, 0, 2])
        img = np.expand_dims(img, axis=0)

        return img

    async def get_result(self, pred, characters: list) -> tuple[str, Any | Literal[1]]:
        accuracy = 1
        last = None
        answer = []
        for item in pred[0]:
            char_ind = item.argmax()
            if char_ind != last and char_ind != 0 and char_ind != len(characters) + 1:
                answer.append(characters[char_ind - 1])
                accuracy *= item[char_ind]
            last = char_ind

        result = "".join(answer)[: self.max_length]

        return result, accuracy

    async def solve_task_ru(self, img) -> list:
        img = await self.extend_ru(img=img)

        result_tensor = self.ModelRu.run(None, {"image": img})[0]
        answer, accuracy = await self.get_result(result_tensor, Chars.characters_ru)

        return [answer, accuracy]

    async def solve_task_en(self, img) -> list:
        img = await self.extend_en(img=img)

        result_tensor = self.ModelEn.run(None, {self.ModelName: img})[0]
        answer, accuracy = await self.get_result(result_tensor, Chars.characters_en)

        return [answer, accuracy]

    async def get_ru_or_en(self, en, ru):
        if en[1] > ru[1]:
            return en[0]
        else:
            return ru[0]

    async def opencv_actions(self, img_arr) -> Any:
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        _, img_bw = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
        img_bw = cv2.bitwise_not(img_bw)

        return img_bw

    async def numpy_actions(self, img_bw) -> np.ndarray:
        im_pil = Image.fromarray(img_bw).convert("L")
        img_arr = np.array(im_pil)

        return img_arr

    async def read(self, file: str | None = None) -> str:
        img = Image.open(file)

        img = img.resize((self.img_width, self.img_height))
        img_arr_en = np.array(img.convert("RGB"))
        img_arr = np.array(img)

        img_bw = await self.opencv_actions(img_arr=img_arr)
        img_arr = await self.numpy_actions(img_bw=img_bw)

        captcha_answer_ru = await self.solve_task_ru(img=img_arr)
        captcha_answer_en = await self.solve_task_en(img=img_arr_en)

        result = await self.get_ru_or_en(captcha_answer_en, captcha_answer_ru)

        return result
