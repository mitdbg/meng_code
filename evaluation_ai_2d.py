from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import torch

def generate_text(model, image_processor, tokenizer, image, caption):

    print("HELLO")

    """
    Step 1: Load images
    """
    # demo_image_one = Image.open(
    #     requests.get(
    #         "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    #     ).raw
    # )

    # demo_image_two = Image.open(
    #     requests.get(
    #         "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
    #         stream=True
    #     ).raw
    # )

    # query_image = Image.open(
    #     requests.get(
    #         "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
    #         stream=True
    #     ).raw
    # )

    print("HELLO")

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    # vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
    vision_x = [image_processor(image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    """
    Step: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    print("HELLO")

    tokenizer.padding_side = "left"
    lang_x = tokenizer(
        # ["<image>" + caption + "<|endofchunk|>"],
        ["<image>What does this image show?"],
        return_tensors="pt",
    )

    print("HELLO")

    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )

    print("HELLO")

    print("Generated text: ", tokenizer.decode(generated_text[0]))
    # """
    # Step 4: Generate text
    # """
    # generated_text = model.generate(
    #     vision_x=vision_x,
    #     lang_x=lang_x["input_ids"],
    #     attention_mask=lang_x["attention_mask"],
    #     max_new_tokens=20,
    #     num_beams=3,
    # )

    # return tokenizer.decode(generated_text[0])

if __name__ == "__main__":

    image = Image.open("../ai2d/images/0.png")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
    )

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    generate_text(model, image_processor, tokenizer, image,  "This is an image of a face")