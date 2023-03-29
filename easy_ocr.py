import cv2
import easyocr
import matplotlib.pyplot as plt

"""""
path = "screen.png"

img = cv2.imread(path)

reader = easyocr.Reader(['en'], gpu=False)

text_ = reader.readtext(img)

for t in text_:
    print(t)
    bbox, text, score = t

    cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 0)

plt.imshow(img)
plt.show()


img_path = "screen.png"
img = cv2.imread(img_path)

reader = easyocr.Reader(['en'], gpu=False)

text_results = reader.readtext(img)

for result in text_results:
    text = result[1]
    bbox = result[0]
    print(text)
    letters = list(text)

    bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]

    cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 0)

    for letter in letters:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 0)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

img_path = "screen.png"
img = cv2.imread(img_path)

reader = easyocr.Reader(['en'], gpu=False)

text_results = reader.readtext(img)

confidence_threshold = 0.1

for result in text_results:
    bbox, text, score = result
    print(result)

    if score >= confidence_threshold:
        bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]

        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)

        letters = list(text)
        for letter in letters:
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
"""""


def text_ocr(img_path, confidence=0.0):
    image = cv2.imread(img_path)
    reader = easyocr.Reader(['en'], gpu=False)

    text_results = reader.readtext(image)

    for result in text_results:
        bbox, text, score = result

        # print(bbox)
        # print(score)
        print(text)

        if score >= confidence:
            bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]

            word = list(text)
            for letters in word:
                cv2.rectangle(image, bbox[0], bbox[2], (255, 255, 0), 1)
    return image


if __name__ == "__main__":
    image = "text2.png"

    result_image = text_ocr(image, confidence=0.1)

    if result_image is not None:
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.show()
