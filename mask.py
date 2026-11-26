import sys

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model.
MODEL = "bert-base-uncased"

# Number of predictions to generate.
K = 3

# Constants used when generating attention diagrams.
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    """Run BERT to predict the masked word and visualize its attention."""
    text = input("Text: ")

    # Tokenize the input sentence.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")

    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Ask BERT to return both predictions and attention weights.
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Print the three most likely replacements for [MASK].
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()

    for token in top_tokens:
        prediction = tokenizer.decode([int(token)]).strip()
        print(text.replace(tokenizer.mask_token, prediction))

    # Generate a diagram for every attention head in every layer.
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """Return the index of the specified mask token, or None if absent."""
    input_ids = inputs["input_ids"][0]

    # TensorFlow tensors need to be converted to ordinary values.
    if hasattr(input_ids, "numpy"):
        input_ids = input_ids.numpy()

    for index, token_id in enumerate(input_ids):
        if int(token_id) == int(mask_token_id):
            return index

    return None


def get_color_for_attention_score(attention_score):
    """Convert an attention score into an RGB grayscale color."""
    if hasattr(attention_score, "numpy"):
        attention_score = attention_score.numpy()

    score = float(attention_score)

    # Keep the value in the valid [0, 1] range.
    score = max(0.0, min(1.0, score))
    gray = round(score * 255)

    return (gray, gray, gray)


def visualize_attentions(tokens, attentions):
    """Generate an attention diagram for every layer and attention head."""
    for layer_index, layer_attention in enumerate(attentions):
        # Each layer contains heads with token-to-token attention scores.
        number_of_heads = len(layer_attention[0])

        for head_index in range(number_of_heads):
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                layer_attention[0][head_index],
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """Generate and save a diagram for one attention head."""
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw token labels along the rows and columns.
    for index, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + index * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + index * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )

    # Shade each cell using the attention score for that token pair.
    for row in range(len(tokens)):
        y = PIXELS_PER_WORD + row * GRID_SIZE

        for column in range(len(tokens)):
            x = PIXELS_PER_WORD + column * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[row][column])
            draw.rectangle(
                (x, y, x + GRID_SIZE, y + GRID_SIZE),
                fill=color,
            )

    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
